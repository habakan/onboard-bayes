/**
 * Reduced-rank GP over a box, by the Hilbert-space method: the Laplace eigenfunctions
 * on a Dirichlet domain, weighted by the kernel's spectral density. An m-term
 * truncation turns a GP into m linear-Gaussian weights, which is what makes the map
 * Rao-Blackwellisable.
 *
 * The magnetic field is curl-free, so it is modelled as B = -∇φ with a GP on the
 * scalar potential φ. The field is then linear in the same weights, with the
 * eigenfunction gradients as its design matrix.
 */

export interface Domain {
  /** Half-extent per axis; the box is [-L, L] on each. */
  L: [number, number, number];
}

export interface Basis {
  m: number;
  domain: Domain;
  /** Eigenvalue λ_j = Σ_d (π n_d / 2L_d)², one per term. */
  lambda: Float64Array;
  /** Integer index triples, one per term. */
  idx: Int32Array;
}

/** The `m` terms with the smallest eigenvalues, which carry the most prior variance. */
export function makeBasis(domain: Domain, m: number): Basis {
  const [Lx, Ly, Lz] = domain.L;
  const cand: { lam: number; n: [number, number, number] }[] = [];
  // Enough per axis that the m smallest are certainly inside the enumerated box.
  const per = Math.max(2, Math.ceil(Math.cbrt(m)) + 3);
  for (let a = 1; a <= per; a++) {
    for (let b = 1; b <= per; b++) {
      for (let c = 1; c <= per; c++) {
        const lam =
          ((Math.PI * a) / (2 * Lx)) ** 2 +
          ((Math.PI * b) / (2 * Ly)) ** 2 +
          ((Math.PI * c) / (2 * Lz)) ** 2;
        cand.push({ lam, n: [a, b, c] });
      }
    }
  }
  cand.sort((p, q) => p.lam - q.lam);
  const keep = cand.slice(0, m);

  const lambda = new Float64Array(m);
  const idx = new Int32Array(m * 3);
  keep.forEach((k, j) => {
    lambda[j] = k.lam;
    idx[j * 3] = k.n[0];
    idx[j * 3 + 1] = k.n[1];
    idx[j * 3 + 2] = k.n[2];
  });
  return { m, domain, lambda, idx };
}

/**
 * Prior variance per weight: the squared-exponential spectral density at √λ, in 3D.
 * `S(ω) = σ² (2π ℓ²)^{3/2} exp(-ℓ²ω²/2)`, so a shorter lengthscale spreads variance
 * over more terms and a longer one concentrates it in the first few.
 */
export function priorVariance(basis: Basis, sigma: number, ell: number): Float64Array {
  const out = new Float64Array(basis.m);
  const scale = sigma * sigma * Math.pow(2 * Math.PI * ell * ell, 1.5);
  for (let j = 0; j < basis.m; j++) {
    out[j] = scale * Math.exp((-ell * ell * basis.lambda[j]!) / 2);
  }
  return out;
}

/** One eigenfunction: `∏_d L_d^{-1/2} sin(π n_d (x_d + L_d) / 2L_d)`. */
export function phi(basis: Basis, j: number, x: readonly number[]): number {
  const { L } = basis.domain;
  let v = 1;
  for (let d = 0; d < 3; d++) {
    const n = basis.idx[j * 3 + d]!;
    const Ld = L[d]!;
    v *= Math.sin((Math.PI * n * (x[d]! + Ld)) / (2 * Ld)) / Math.sqrt(Ld);
  }
  return v;
}

/**
 * `H[d][j] = -∂φ_j/∂x_d`, the design matrix mapping weights to a field measurement.
 * The sign is the one in `B = -∇φ`, so the caller can treat this as an ordinary
 * linear observation.
 */
export function fieldDesign(basis: Basis, x: readonly number[], out: Float64Array): void {
  const { L } = basis.domain;
  for (let j = 0; j < basis.m; j++) {
    const s: number[] = [0, 0, 0];
    const c: number[] = [0, 0, 0];
    const k: number[] = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const n = basis.idx[j * 3 + d]!;
      const Ld = L[d]!;
      k[d] = (Math.PI * n) / (2 * Ld);
      const arg = k[d]! * (x[d]! + Ld);
      const inv = 1 / Math.sqrt(Ld);
      s[d] = Math.sin(arg) * inv;
      c[d] = Math.cos(arg) * inv;
    }
    // Differentiating the product swaps sin for k·cos on the differentiated axis.
    out[0 * basis.m + j] = -(k[0]! * c[0]! * s[1]! * s[2]!);
    out[1 * basis.m + j] = -(s[0]! * k[1]! * c[1]! * s[2]!);
    out[2 * basis.m + j] = -(s[0]! * s[1]! * k[2]! * c[2]!);
  }
}
