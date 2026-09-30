#include "bw_syn/transforms.hpp"

#include "bw_syn/detail/ir_build.hpp"

#include <vector>

namespace bw_syn {

bool has_intermediate_bf16_round(const Program& p) {
  if (p.result < 0)
    return false;
  std::vector<char> seen(p.nodes.size(), 0);
  std::vector<int> stack{p.result};
  while (!stack.empty()) {
    const int i = stack.back();
    stack.pop_back();
    if (i < 0 || i >= static_cast<int>(p.nodes.size()) || seen[static_cast<size_t>(i)])
      continue;
    seen[static_cast<size_t>(i)] = 1;
    const auto& n = p.nodes[static_cast<size_t>(i)];
    if (n.op == OpKind::RoundBF16 && i != p.result)
      return true;
    for (int a : n.args)
      stack.push_back(a);
  }
  return false;
}

namespace {

bool is_input_g(const Program& p, int idx) {
  return idx >= 0 && idx < static_cast<int>(p.nodes.size()) &&
         p.nodes[static_cast<size_t>(idx)].op == OpKind::InputG;
}

int unwrap_round(const Program& p, int idx) {
  if (idx < 0 || idx >= static_cast<int>(p.nodes.size()))
    return idx;
  const auto& n = p.nodes[static_cast<size_t>(idx)];
  return (n.op == OpKind::RoundBF16 && !n.args.empty()) ? n.args[0] : idx;
}

struct MulPattern {
  int mul_idx{-1};
  int g_idx{-1};
  int d_idx{-1};
  bool d_was_rounded{false};
};

MulPattern find_g_mul_deriv(const Program& p) {
  MulPattern pat;
  int root = p.result;
  if (root < 0)
    return pat;
  if (p.nodes[static_cast<size_t>(root)].op == OpKind::RoundBF16 &&
      !p.nodes[static_cast<size_t>(root)].args.empty())
    root = p.nodes[static_cast<size_t>(root)].args[0];
  if (root < 0 || root >= static_cast<int>(p.nodes.size()))
    return pat;
  const auto& mul = p.nodes[static_cast<size_t>(root)];
  if (mul.op != OpKind::Mul || mul.args.size() != 2)
    return pat;
  const int a = mul.args[0];
  const int b = mul.args[1];
  if (is_input_g(p, a)) {
    pat.g_idx = a;
    pat.d_was_rounded = p.nodes[static_cast<size_t>(b)].op == OpKind::RoundBF16;
    pat.d_idx = unwrap_round(p, b);
  } else if (is_input_g(p, b)) {
    pat.g_idx = b;
    pat.d_was_rounded = p.nodes[static_cast<size_t>(a)].op == OpKind::RoundBF16;
    pat.d_idx = unwrap_round(p, a);
  } else {
    return pat;
  }
  pat.mul_idx = root;
  return pat;
}

Program rewrite_scale_sep(const Program& input, const MulPattern& pat) {
  Program p = input;
  p.name = input.name.empty() ? "scale_separated" : input.name + "_scale_sep";
  p.result = detail::add_scale_product(p, pat.g_idx, pat.d_idx);
  return p;
}

} // namespace

TransformResult apply_scale_separate_mul(const Program& input) {
  TransformResult tr;
  tr.kind = TransformKind::ScaleSeparateMul;
  tr.program = input;
  tr.preconditions.no_prior_bf16_of_derivative = true;

  const auto pat = find_g_mul_deriv(input);
  if (pat.mul_idx < 0)
    return tr;
  if (pat.d_was_rounded)
    tr.preconditions.no_prior_bf16_of_derivative = false;

  tr.program = rewrite_scale_sep(input, pat);
  tr.applied = true;
  tr.sound_under_preconditions =
      tr.preconditions.no_prior_bf16_of_derivative && !has_intermediate_bf16_round(tr.program);
  return tr;
}

TransformResult apply_insert_normalize(const Program& input) {
  TransformResult tr;
  tr.kind = TransformKind::InsertNormalize;
  tr.program = input;

  int rec = -1;
  for (int i = 0; i < static_cast<int>(input.nodes.size()); ++i)
    if (input.nodes[static_cast<size_t>(i)].op == OpKind::Reconstruct)
      rec = i;
  if (rec < 0 || input.nodes[static_cast<size_t>(rec)].args.size() != 2)
    return tr;

  const auto& rn = input.nodes[static_cast<size_t>(rec)];
  for (const auto& n : input.nodes) {
    if (n.op == OpKind::Normalize && n.args.size() == 2 && n.args[0] == rn.args[0] &&
        n.args[1] == rn.args[1]) {
      // Structural: Normalize already feeds Reconstruct; not a ScaleSeparateMul proof.
      tr.sound_under_preconditions = true;
      return tr;
    }
  }

  Program p = input;
  const int nm = p.add(Node{OpKind::Normalize, {rn.args[0], rn.args[1]}});
  const int m2 = p.add(Node{OpKind::FrexpMant, {nm}});
  const int e2 = p.add(Node{OpKind::FrexpExp, {nm}});
  const int new_rec = p.add(Node{OpKind::Reconstruct, {m2, e2}});
  if (p.result == rec ||
      (p.result >= 0 && p.nodes[static_cast<size_t>(p.result)].op == OpKind::RoundBF16 &&
       !p.nodes[static_cast<size_t>(p.result)].args.empty() &&
       p.nodes[static_cast<size_t>(p.result)].args[0] == rec))
    p.result = p.add(Node{OpKind::RoundBF16, {new_rec}});
  else
    p.result = new_rec;

  tr.program = p;
  tr.applied = true;
  tr.sound_under_preconditions = true;
  return tr;
}

TransformResult apply_defer_rounding(const Program& input) {
  TransformResult tr;
  tr.kind = TransformKind::DeferRounding;
  tr.program = input;

  const auto pat = find_g_mul_deriv(input);
  if (pat.mul_idx < 0 || !pat.d_was_rounded) {
    tr.applied = !has_intermediate_bf16_round(input);
    tr.sound_under_preconditions = tr.applied;
    return tr;
  }

  Program p = input;
  int root = p.result;
  if (p.nodes[static_cast<size_t>(root)].op == OpKind::RoundBF16)
    root = p.nodes[static_cast<size_t>(root)].args[0];
  auto& mul = p.nodes[static_cast<size_t>(root)];
  if (mul.args[0] == pat.g_idx)
    mul.args[1] = pat.d_idx;
  else
    mul.args[0] = pat.d_idx;

  auto sep = apply_scale_separate_mul(p);
  sep.kind = TransformKind::DeferRounding;
  return sep;
}

TransformResult apply_default_scale_pipeline(const Program& input) {
  auto a = apply_defer_rounding(input);
  Program cur = a.applied ? a.program : input;
  if (!a.applied) {
    a = apply_scale_separate_mul(cur);
    if (a.applied)
      cur = a.program;
  }
  const auto b = apply_insert_normalize(cur);
  TransformResult c;
  c.kind = TransformKind::DefaultScalePipeline;
  c.program = b.applied ? b.program : cur;
  c.applied = a.applied || b.applied;
  c.preconditions = a.preconditions;
  c.sound_under_preconditions =
      a.sound_under_preconditions && (!b.applied || b.sound_under_preconditions);
  return c;
}

} // namespace bw_syn
