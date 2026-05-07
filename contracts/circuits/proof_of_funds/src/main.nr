// ═══════════════════════════════════════════════════════════════════
// PropOS ZKP Circuit — Zero-Knowledge Proof of Funds
// ═══════════════════════════════════════════════════════════════════
//
// Purpose:
//   A buyer proves they hold assets ≥ threshold WITHOUT revealing
//   their actual balance. The proof is ~300 bytes (zk-SNARK/PLONK)
//   and verifiable on-chain for ~200k gas.
//
// Public inputs:
//   - threshold:  minimum balance required (e.g., $500,000)
//   - commitment: H(balance || salt) — published on-chain
//
// Private inputs (witness — never revealed):
//   - balance:    actual account balance
//   - salt:       random blinding factor
//
// The circuit proves:
//   1. balance ≥ threshold              (range proof)
//   2. H(balance || salt) == commitment  (commitment integrity)
//
// Compiled to ACIR → Barretenberg backend → PLONK proof
// ═══════════════════════════════════════════════════════════════════

use std::hash::pedersen_hash;

// ── Main circuit entry point ─────────────────────────────────────

fn main(
    // Public inputs (known to verifier)
    threshold: pub Field,
    commitment: pub Field,

    // Private inputs (witness — known only to prover)
    balance: Field,
    salt: Field,
) {
    // ── Constraint 1: Range Proof ────────────────────────────────
    // Prove that balance ≥ threshold without revealing balance.
    //
    // In Noir, Field comparisons use the `assert` keyword which
    // creates an R1CS constraint. The subtraction ensures the
    // result is non-negative (wraps in finite field if negative,
    // causing the proof to fail).

    let surplus = balance - threshold;

    // Assert surplus fits in 64 bits (i.e., is non-negative and bounded).
    // This is the range proof: if balance < threshold, the subtraction
    // wraps around in the field and cannot fit in 64 bits.
    assert(surplus as u64 as Field == surplus);

    // ── Constraint 2: Commitment Integrity ───────────────────────
    // Prove that the public commitment matches H(balance || salt).
    // This binds the proof to a specific balance without revealing it.
    //
    // Uses Pedersen hash (algebraic, SNARK-friendly, ~2000 constraints).
    let computed_commitment = pedersen_hash([balance, salt]);
    assert(computed_commitment == commitment);
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn test_valid_proof() {
    // Buyer has $750,000, needs to prove ≥ $500,000
    let balance: Field = 750000;
    let salt: Field = 12345678;
    let threshold: Field = 500000;
    let commitment = pedersen_hash([balance, salt]);

    main(threshold, commitment, balance, salt);
    // If we get here without panic, the proof is valid
}

#[test(should_fail)]
fn test_insufficient_funds() {
    // Buyer has $400,000 but claims ≥ $500,000 — should fail
    let balance: Field = 400000;
    let salt: Field = 87654321;
    let threshold: Field = 500000;
    let commitment = pedersen_hash([balance, salt]);

    main(threshold, commitment, balance, salt);
}

#[test(should_fail)]
fn test_wrong_commitment() {
    // Buyer tries to use a different balance for the commitment
    let real_balance: Field = 750000;
    let fake_balance: Field = 300000;
    let salt: Field = 11111111;
    let threshold: Field = 500000;

    // Commitment made with fake balance
    let commitment = pedersen_hash([fake_balance, salt]);

    // But proving with real balance — commitment mismatch should fail
    main(threshold, commitment, real_balance, salt);
}

#[test]
fn test_exact_threshold() {
    // Edge case: balance exactly equals threshold
    let balance: Field = 500000;
    let salt: Field = 99999999;
    let threshold: Field = 500000;
    let commitment = pedersen_hash([balance, salt]);

    main(threshold, commitment, balance, salt);
}
