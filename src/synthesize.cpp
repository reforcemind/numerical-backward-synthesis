#include "bw_syn/synthesize.hpp"

#include "bw_syn/detail/scale_eval.hpp"
#include "bw_syn/functions/elu_bw.hpp"
#include "bw_syn/functions/erf_bw.hpp"
#include "bw_syn/functions/sigmoid_bw.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/transforms.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace bw_syn {

std::string synth_params_key(const SynthParams& p) {
  std::ostringstream os;
  os << "rr" << static_cast<int>(p.range_red) << "_ev" << static_cast<int>(p.eval) << "_nm"
     << static_cast<int>(p.norm) << "_rc" << static_cast<int>(p.reconstruct) << "_rd"
     << static_cast<int>(p.round);
  return os.str();
}

static std::string reachable_program_key(const Program& program) {
  const auto live = reachable_nodes(program);
  std::vector<int> canonical(program.nodes.size(), -1);
  std::ostringstream os;
  for (size_t position = 0; position < live.size(); ++position) {
    const int i = live[position];
    const auto& node = program.nodes[static_cast<size_t>(i)];
    canonical[static_cast<size_t>(i)] = static_cast<int>(position);
    os << static_cast<int>(node.op);
    if (node.op == OpKind::ConstF64)
      os << ':' << std::hexfloat << node.c_f64 << std::defaultfloat;
    if (node.op == OpKind::ConstI32)
      os << ':' << node.c_i32;
    os << '(';
    for (int arg : node.args)
      os << canonical[static_cast<size_t>(arg)] << ',';
    os << ");";
  }
  return os.str();
}

static Program base_program(BackwardKind kind, double alpha) {
  if (kind == BackwardKind::Tanh)
    return tanh_bw::ir_direct();
  if (kind == BackwardKind::Sigmoid)
    return sigmoid_bw::ir_direct();
  if (kind == BackwardKind::Erf)
    return erf_bw::ir_direct();
  return elu_bw::ir_direct(alpha);
}

static void mutate_round_mant(Program& p) {
  for (int i = 0; i < static_cast<int>(p.nodes.size()); ++i) {
    if (p.nodes[static_cast<size_t>(i)].op != OpKind::Reconstruct ||
        p.nodes[static_cast<size_t>(i)].args.size() != 2)
      continue;
    const int m = p.nodes[static_cast<size_t>(i)].args[0];
    const int e = p.nodes[static_cast<size_t>(i)].args[1];
    const int mr = p.add(Node{OpKind::RoundBF16, {m}});
    p.nodes[static_cast<size_t>(i)].args = {mr, e};
    p.name += "_rtl";
    return;
  }
}

static void mutate_mid_round(Program& p) {
  for (int i = 0; i < static_cast<int>(p.nodes.size()); ++i) {
    if (p.nodes[static_cast<size_t>(i)].op != OpKind::Frexp ||
        p.nodes[static_cast<size_t>(i)].args.empty())
      continue;
    const int d = p.nodes[static_cast<size_t>(i)].args[0];
    if (d < 0 || p.nodes[static_cast<size_t>(d)].op == OpKind::InputG ||
        p.nodes[static_cast<size_t>(d)].op == OpKind::RoundBF16)
      continue;
    const int dr = p.add(Node{OpKind::RoundBF16, {d}});
    p.nodes[static_cast<size_t>(i)].args[0] = dr;
    return;
  }
}

static void strip_normalize(Program& p) {
  for (auto& n : p.nodes) {
    if (n.op != OpKind::Reconstruct || n.args.size() != 2)
      continue;
    const int a = n.args[0];
    if (a < 0 || p.nodes[static_cast<size_t>(a)].op != OpKind::FrexpMant)
      continue;
    const int nm =
        p.nodes[static_cast<size_t>(a)].args.empty() ? -1 : p.nodes[static_cast<size_t>(a)].args[0];
    if (nm < 0 || p.nodes[static_cast<size_t>(nm)].op != OpKind::Normalize ||
        p.nodes[static_cast<size_t>(nm)].args.size() != 2)
      continue;
    n.args = p.nodes[static_cast<size_t>(nm)].args;
  }
}

static void clamp_abs_x(Program& p) {
  int x = -1;
  for (int i = 0; i < static_cast<int>(p.nodes.size()); ++i)
    if (p.nodes[static_cast<size_t>(i)].op == OpKind::InputX)
      x = i;
  if (x < 0)
    return;
  for (int i = 0; i < static_cast<int>(p.nodes.size()); ++i) {
    if (p.nodes[static_cast<size_t>(i)].op != OpKind::Abs ||
        p.nodes[static_cast<size_t>(i)].args.empty() ||
        p.nodes[static_cast<size_t>(i)].args[0] != x)
      continue;
    const int lo = p.add(Node{OpKind::ConstF64, {}, 0.0});
    const int hi = p.add(Node{OpKind::ConstF64, {}, 80.0});
    const int c = p.add(Node{OpKind::Clamp, {x, lo, hi}});
    p.nodes[static_cast<size_t>(i)].args[0] = c;
    return;
  }
}

static void preserve_infinite_gradient(Program& p) {
  int x = -1;
  int g = -1;
  for (int i = 0; i < static_cast<int>(p.nodes.size()); ++i) {
    if (p.nodes[static_cast<size_t>(i)].op == OpKind::InputX)
      x = i;
    if (p.nodes[static_cast<size_t>(i)].op == OpKind::InputG)
      g = i;
  }
  if (x < 0 || g < 0)
    throw std::invalid_argument("backward program needs x and g inputs");
  p.result = p.add(Node{OpKind::PositiveDerivativeInf, {x, g, p.result}});
}

Program build_candidate_program(BackwardKind kind, const SynthParams& p, double alpha) {
  if (p.eval == EvalStrategy::TailSplit4) {
    if (kind != BackwardKind::Tanh)
      throw std::invalid_argument("TailSplit4 currently supports tanh backward only");
    return tanh_bw::ir_tail_split4();
  }
  Program prog = base_program(kind, alpha);
  if (p.range_red == RangeReduction::ClampDomain)
    clamp_abs_x(prog);
  if (p.eval == EvalStrategy::ScaleSeparated) {
    const auto tr = apply_default_scale_pipeline(prog);
    if (!tr.applied)
      throw std::logic_error("scale transform did not match backward product");
    prog = tr.program;
    if (p.round == RoundStrategy::IntermediateAndFinal)
      mutate_mid_round(prog);

    if (p.norm == NormStrategy::None)
      strip_normalize(prog);
    else {
      auto n = apply_insert_normalize(prog);
      if (n.applied)
        prog = n.program;
    }

    if (p.reconstruct == ReconstructStrategy::RoundThenLdexp)
      mutate_round_mant(prog);
  }
  if (kind == BackwardKind::Tanh || kind == BackwardKind::Sigmoid || kind == BackwardKind::Erf)
    preserve_infinite_gradient(prog);
  return prog;
}

SynthResult synthesize(const SynthConfig& cfg) {
  SynthResult out;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<SynthParams> space;

  if (cfg.full_grammar) {
    // AbsSymmetry ≡ None for current builders; AfterMul ≡ BeforeMul — omit no-op axes.
    for (auto rr : {RangeReduction::None, RangeReduction::ClampDomain})
      for (auto ev : {EvalStrategy::DirectMaterialize, EvalStrategy::ScaleSeparated})
        for (auto nm : {NormStrategy::None, NormStrategy::BeforeMul})
          for (auto rc : {ReconstructStrategy::LdexpThenRound, ReconstructStrategy::RoundThenLdexp})
            for (auto rd : {RoundStrategy::FinalOnly, RoundStrategy::IntermediateAndFinal}) {
              SynthParams p;
              p.range_red = rr;
              p.eval = ev;
              p.norm = nm;
              p.reconstruct = rc;
              p.round = rd;
              space.push_back(p);
            }
  } else {
    SynthParams good;
    good.eval = EvalStrategy::ScaleSeparated;
    space.push_back(good);
    SynthParams bad = good;
    bad.eval = EvalStrategy::DirectMaterialize;
    bad.round = RoundStrategy::IntermediateAndFinal;
    space.push_back(bad);
  }

  if (cfg.kind == BackwardKind::Tanh && cfg.contract.min_abs_x && *cfg.contract.min_abs_x >= 4.0 &&
      cfg.contract.finite_inputs_only && cfg.contract.normal_reference_output_only) {
    SynthParams tail;
    tail.eval = EvalStrategy::TailSplit4;
    space.push_back(tail);
  }

  out.search_space_size = space.size();
  std::unordered_set<std::string> seen_programs;
  for (const auto& params : space) {
    Candidate cand;
    cand.params = params;
    cand.id = synth_params_key(params);
    cand.program = build_candidate_program(cfg.kind, params, cfg.alpha);
    if (!seen_programs.insert(reachable_program_key(cand.program)).second)
      continue;
    cand.cost = estimate_cost(cand.program, cfg.cost_model);
    ++out.candidates_built;
    VerifyConfig vcfg = cfg.verify_cfg;
    vcfg.kind = cfg.kind;
    vcfg.contract = cfg.contract;
    vcfg.alpha = cfg.alpha;
    cand.verify = verify_kernel(
        [&](BF16 x, BF16 g) {
          return detail::eval_ir_under_contract(cand.program, x, g, cfg.contract);
        },
        vcfg);
    cand.verified = cand.verify.ok && cand.verify.counters.max_ulp <= cfg.max_ulp_to_accept &&
                    cand.verify.counters.false_zeros == 0;
    if (cand.verified)
      ++out.candidates_verified;
    out.all.push_back(std::move(cand));
  }

  std::vector<Candidate*> pool;
  for (auto& c : out.all)
    if (c.verified)
      pool.push_back(&c);

  for (auto* a : pool) {
    const double ua = a->verify.counters.max_ulp + 1000.0 * a->verify.counters.false_zeros;
    bool dominated = false;
    for (auto* b : pool) {
      if (a == b)
        continue;
      const double ub = b->verify.counters.max_ulp + 1000.0 * b->verify.counters.false_zeros;
      if (dominates(ub, b->cost.scalar_score, ua, a->cost.scalar_score)) {
        dominated = true;
        break;
      }
    }
    if (!dominated)
      out.pareto.push_back(*a);
  }

  double best = 1e300;
  for (auto& c : out.all) {
    if (c.verified && c.cost.scalar_score < best) {
      best = c.cost.scalar_score;
      out.best = c;
      out.found = true;
    }
  }
  out.synthesis_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.summary = out.found ? out.best.id : "none";
  return out;
}

} // namespace bw_syn
