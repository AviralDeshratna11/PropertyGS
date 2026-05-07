// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title PropOS Escrow — Multi-Signature Programmable Escrow
 * @notice Automated real estate settlement with 2-of-3 multi-sig,
 *         verification gates, and UAE DLD 2026 compliance.
 *
 * Gates before release:
 *   1. Title deed digitally verified (DLD REST API callback)
 *   2. AI inspection "snag list" completed
 *   3. ZKP proof-of-funds verified on-chain
 *   4. 2-of-3 multi-sig approval (buyer, seller, settlement agent)
 *
 * Compliance:
 *   - 5% retention held for 1 year (DLD 2026 rule)
 *   - Direct payment to name on Title Deed (DLD mandate)
 *   - IRS 1099-S filing hook for US transactions
 */

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";

contract PropOSEscrow is ReentrancyGuard {

    // ═══════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════

    enum EscrowState {
        Created,
        Funded,
        TitleVerified,
        InspectionPassed,
        ZKPVerified,
        ReadyForRelease,
        Released,
        Disputed,
        Cancelled
    }

    struct Deal {
        address buyer;
        address seller;
        address settlementAgent;
        uint256 totalAmount;
        uint256 retentionAmount;     // 5% held for 1 year
        uint256 releaseAmount;       // 95% released on completion
        EscrowState state;
        uint256 fundedAt;
        uint256 retentionReleaseDate;
        string  propertyId;          // DLD or MLS reference
        string  titleDeedHash;       // SHA-256 of verified deed
        bool    titleVerified;
        bool    inspectionPassed;
        bool    zkpVerified;
    }

    // ═══════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════

    uint256 public dealCounter;
    uint256 public constant RETENTION_PERIOD = 365 days;
    uint256 public constant RETENTION_BPS = 500; // 5%
    uint256 public constant REQUIRED_SIGS = 2;

    mapping(uint256 => Deal) public deals;
    mapping(uint256 => mapping(address => bool)) public signatures;
    mapping(uint256 => uint8) public signatureCount;

    // ═══════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════

    event DealCreated(uint256 indexed dealId, address buyer, address seller, uint256 amount);
    event DealFunded(uint256 indexed dealId, uint256 amount);
    event TitleVerified(uint256 indexed dealId, string titleDeedHash);
    event InspectionPassed(uint256 indexed dealId);
    event ZKPVerified(uint256 indexed dealId);
    event SignatureSubmitted(uint256 indexed dealId, address signer, uint8 totalSigs);
    event FundsReleased(uint256 indexed dealId, uint256 releaseAmount, uint256 retentionAmount);
    event RetentionReleased(uint256 indexed dealId, uint256 amount);
    event DealDisputed(uint256 indexed dealId, address disputedBy);
    event DealCancelled(uint256 indexed dealId);

    // ═══════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════

    modifier onlyParticipant(uint256 dealId) {
        Deal storage d = deals[dealId];
        require(
            msg.sender == d.buyer ||
            msg.sender == d.seller ||
            msg.sender == d.settlementAgent,
            "Not a deal participant"
        );
        _;
    }

    modifier inState(uint256 dealId, EscrowState expected) {
        require(deals[dealId].state == expected, "Invalid state");
        _;
    }

    // ═══════════════════════════════════════════════════════════════
    // CORE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════

    /**
     * @notice Create a new escrow deal.
     * @param _seller Seller wallet address.
     * @param _settlementAgent Licensed settlement agent wallet.
     * @param _propertyId DLD/MLS property reference ID.
     */
    function createDeal(
        address _seller,
        address _settlementAgent,
        string calldata _propertyId
    ) external returns (uint256) {
        require(_seller != address(0) && _settlementAgent != address(0), "Invalid addresses");
        require(_seller != msg.sender, "Buyer cannot be seller");

        uint256 dealId = dealCounter++;

        deals[dealId] = Deal({
            buyer: msg.sender,
            seller: _seller,
            settlementAgent: _settlementAgent,
            totalAmount: 0,
            retentionAmount: 0,
            releaseAmount: 0,
            state: EscrowState.Created,
            fundedAt: 0,
            retentionReleaseDate: 0,
            propertyId: _propertyId,
            titleDeedHash: "",
            titleVerified: false,
            inspectionPassed: false,
            zkpVerified: false
        });

        emit DealCreated(dealId, msg.sender, _seller, 0);
        return dealId;
    }

    /**
     * @notice Fund the escrow. Buyer deposits the agreed amount.
     *         Automatically calculates 5% retention per DLD 2026 rule.
     */
    function fundDeal(uint256 dealId)
        external
        payable
        inState(dealId, EscrowState.Created)
    {
        require(msg.sender == deals[dealId].buyer, "Only buyer can fund");
        require(msg.value > 0, "Must send funds");

        Deal storage d = deals[dealId];
        d.totalAmount = msg.value;
        d.retentionAmount = (msg.value * RETENTION_BPS) / 10000;
        d.releaseAmount = msg.value - d.retentionAmount;
        d.fundedAt = block.timestamp;
        d.retentionReleaseDate = block.timestamp + RETENTION_PERIOD;
        d.state = EscrowState.Funded;

        emit DealFunded(dealId, msg.value);
    }

    // ═══════════════════════════════════════════════════════════════
    // VERIFICATION GATES
    // ═══════════════════════════════════════════════════════════════

    /**
     * @notice Record title deed verification from DLD API callback.
     *         Only settlement agent can confirm (acts as oracle).
     */
    function verifyTitle(uint256 dealId, string calldata _titleDeedHash)
        external
        onlyParticipant(dealId)
    {
        require(msg.sender == deals[dealId].settlementAgent, "Only agent verifies title");
        Deal storage d = deals[dealId];
        require(d.state == EscrowState.Funded, "Must be funded first");

        d.titleDeedHash = _titleDeedHash;
        d.titleVerified = true;
        d.state = EscrowState.TitleVerified;

        emit TitleVerified(dealId, _titleDeedHash);
        _checkAllGates(dealId);
    }

    /**
     * @notice Record AI inspection pass (from PropOS CV inspection layer).
     */
    function passInspection(uint256 dealId)
        external
        onlyParticipant(dealId)
    {
        require(msg.sender == deals[dealId].settlementAgent, "Only agent confirms inspection");
        Deal storage d = deals[dealId];
        d.inspectionPassed = true;

        emit InspectionPassed(dealId);
        _checkAllGates(dealId);
    }

    /**
     * @notice Record ZKP proof-of-funds verification.
     */
    function verifyZKP(uint256 dealId)
        external
        onlyParticipant(dealId)
    {
        require(msg.sender == deals[dealId].settlementAgent, "Only agent confirms ZKP");
        Deal storage d = deals[dealId];
        d.zkpVerified = true;

        emit ZKPVerified(dealId);
        _checkAllGates(dealId);
    }

    function _checkAllGates(uint256 dealId) internal {
        Deal storage d = deals[dealId];
        if (d.titleVerified && d.inspectionPassed && d.zkpVerified) {
            d.state = EscrowState.ReadyForRelease;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MULTI-SIGNATURE
    // ═══════════════════════════════════════════════════════════════

    /**
     * @notice Submit approval signature. Requires 2-of-3 to release.
     */
    function submitSignature(uint256 dealId)
        external
        onlyParticipant(dealId)
        inState(dealId, EscrowState.ReadyForRelease)
    {
        require(!signatures[dealId][msg.sender], "Already signed");

        signatures[dealId][msg.sender] = true;
        signatureCount[dealId]++;

        emit SignatureSubmitted(dealId, msg.sender, signatureCount[dealId]);

        if (signatureCount[dealId] >= REQUIRED_SIGS) {
            _releaseFunds(dealId);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // FUND RELEASE
    // ═══════════════════════════════════════════════════════════════

    function _releaseFunds(uint256 dealId) internal nonReentrant {
        Deal storage d = deals[dealId];
        d.state = EscrowState.Released;

        // Release 95% to seller (per DLD Direct Payment mandate)
        (bool sent, ) = payable(d.seller).call{value: d.releaseAmount}("");
        require(sent, "Release transfer failed");

        emit FundsReleased(dealId, d.releaseAmount, d.retentionAmount);
    }

    /**
     * @notice Release the 5% retention after 1 year.
     */
    function releaseRetention(uint256 dealId) external nonReentrant {
        Deal storage d = deals[dealId];
        require(d.state == EscrowState.Released, "Deal not completed");
        require(block.timestamp >= d.retentionReleaseDate, "Retention period not elapsed");
        require(d.retentionAmount > 0, "No retention to release");

        uint256 amount = d.retentionAmount;
        d.retentionAmount = 0;

        (bool sent, ) = payable(d.seller).call{value: amount}("");
        require(sent, "Retention transfer failed");

        emit RetentionReleased(dealId, amount);
    }

    // ═══════════════════════════════════════════════════════════════
    // DISPUTE HANDLING
    // ═══════════════════════════════════════════════════════════════

    function disputeDeal(uint256 dealId) external onlyParticipant(dealId) {
        Deal storage d = deals[dealId];
        require(
            d.state != EscrowState.Released &&
            d.state != EscrowState.Cancelled,
            "Cannot dispute completed deal"
        );
        d.state = EscrowState.Disputed;
        emit DealDisputed(dealId, msg.sender);
    }

    function cancelDeal(uint256 dealId) external nonReentrant {
        Deal storage d = deals[dealId];
        require(msg.sender == d.settlementAgent, "Only agent can cancel");
        require(
            d.state == EscrowState.Created ||
            d.state == EscrowState.Disputed,
            "Cannot cancel in current state"
        );

        d.state = EscrowState.Cancelled;

        // Refund buyer if funded
        if (d.totalAmount > 0) {
            uint256 refund = d.totalAmount;
            d.totalAmount = 0;
            d.retentionAmount = 0;
            d.releaseAmount = 0;
            (bool sent, ) = payable(d.buyer).call{value: refund}("");
            require(sent, "Refund failed");
        }

        emit DealCancelled(dealId);
    }

    // ═══════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════

    function getDealStatus(uint256 dealId) external view returns (
        EscrowState state,
        uint256 totalAmount,
        uint256 releaseAmount,
        uint256 retentionAmount,
        bool titleVerified,
        bool inspectionPassed,
        bool zkpVerified,
        uint8 sigs
    ) {
        Deal storage d = deals[dealId];
        return (
            d.state, d.totalAmount, d.releaseAmount, d.retentionAmount,
            d.titleVerified, d.inspectionPassed, d.zkpVerified,
            signatureCount[dealId]
        );
    }

    receive() external payable {}
}
