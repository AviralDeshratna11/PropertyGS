// ═══════════════════════════════════════════════════════════════════
// PropOS ZKP — Circom Circuit for Proof of Funds
// ═══════════════════════════════════════════════════════════════════
// Alternative to the Noir implementation using Circom + snarkjs.
// Requires deeper understanding of R1CS and finite field arithmetic.
//
// Compile: circom proof_of_funds.circom --r1cs --wasm --sym
// Setup:   snarkjs groth16 setup proof_of_funds.r1cs pot_final.ptau circuit.zkey
// Prove:   snarkjs groth16 prove circuit.zkey witness.wtns proof.json public.json
// Verify:  snarkjs groth16 verify verification_key.json public.json proof.json
// ═══════════════════════════════════════════════════════════════════

pragma circom 2.1.6;

include "circomlib/poseidon.circom";
include "circomlib/comparators.circom";
include "circomlib/bitify.circom";

// Range proof: proves balance >= threshold without revealing balance
template RangeCheck(n) {
    signal input value;
    signal input max_value;

    // Decompose (value - max_value + 2^n) into bits
    // If value >= max_value, the result fits in n bits
    signal input diff;
    diff <== value - max_value + (1 << n);

    component bits = Num2Bits(n + 1);
    bits.in <== diff;

    // The top bit must be 1 (proves non-negative after offset)
    bits.out[n] === 1;
}

// Main circuit: Proof of Funds
template ProofOfFunds() {
    // ── Public inputs ────────────────────────────────────────
    signal input threshold;       // Minimum balance required
    signal input commitment;      // Poseidon(balance, salt) — published on-chain

    // ── Private inputs (witness) ─────────────────────────────
    signal input balance;         // Actual balance (NEVER revealed)
    signal input salt;            // Random blinding factor

    // ── Constraint 1: Range Proof ────────────────────────────
    // Prove balance >= threshold using 64-bit decomposition
    component rangeCheck = RangeCheck(64);
    rangeCheck.value <== balance;
    rangeCheck.max_value <== threshold;

    // Surplus must be non-negative (handled by RangeCheck)
    signal surplus;
    surplus <== balance - threshold;

    // ── Constraint 2: Commitment Integrity ───────────────────
    // Prove Poseidon(balance, salt) == commitment
    component hasher = Poseidon(2);
    hasher.inputs[0] <== balance;
    hasher.inputs[1] <== salt;

    // Enforce equality
    commitment === hasher.out;
}

// Instantiate
component main {public [threshold, commitment]} = ProofOfFunds();

// ═══════════════════════════════════════════════════════════════════
// INPUT EXAMPLE (input.json):
// {
//   "threshold": "500000",
//   "commitment": "<poseidon_hash_output>",
//   "balance": "750000",
//   "salt": "12345678"
// }
// ═══════════════════════════════════════════════════════════════════
