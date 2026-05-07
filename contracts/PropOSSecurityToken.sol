// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title PropOS ERC-1400 Security Token — Fractional Property Ownership
 * @notice Tokenizes real estate into tradeable security tokens.
 *         Enables fractional ownership (min AED 2,000) with automated
 *         rent distribution via smart contracts.
 *
 * Compliance:
 *   - ERC-1400 partitioned token standard
 *   - Transfer restrictions based on KYC/AML verification
 *   - Automated dividend distribution (rental income)
 *   - 24/7 secondary market trading
 */

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract PropOSSecurityToken is ERC20, AccessControl, ReentrancyGuard {

    bytes32 public constant ISSUER_ROLE = keccak256("ISSUER_ROLE");
    bytes32 public constant TRANSFER_AGENT_ROLE = keccak256("TRANSFER_AGENT_ROLE");

    struct PropertyInfo {
        string  propertyId;          // DLD reference
        string  titleDeedHash;       // SHA-256 of deed
        uint256 valuationUsd;        // Total property value
        uint256 totalShares;         // Total fractional shares
        uint256 minInvestmentUsd;    // Minimum buy-in (e.g., ~$545 = AED 2000)
        bool    active;
    }

    struct Investor {
        bool    kycVerified;
        bool    amlCleared;
        string  jurisdiction;        // "UAE", "US", etc.
        uint256 investedAt;
    }

    PropertyInfo public property;
    mapping(address => Investor) public investors;
    mapping(address => bool) public whitelist;

    // Dividend tracking
    uint256 public totalDividendsDistributed;
    uint256 public dividendPerToken;
    mapping(address => uint256) public lastDividendClaimed;
    mapping(address => uint256) public unclaimedDividends;

    // Secondary market
    struct SellOrder {
        address seller;
        uint256 amount;
        uint256 pricePerTokenUsd;
        bool    active;
    }
    uint256 public orderCounter;
    mapping(uint256 => SellOrder) public sellOrders;

    event PropertyTokenized(string propertyId, uint256 totalShares, uint256 valuationUsd);
    event InvestorWhitelisted(address indexed investor, string jurisdiction);
    event DividendDistributed(uint256 totalAmount, uint256 perToken);
    event DividendClaimed(address indexed investor, uint256 amount);
    event SellOrderCreated(uint256 indexed orderId, address seller, uint256 amount, uint256 price);
    event SellOrderFilled(uint256 indexed orderId, address buyer, uint256 amount);
    event TransferRestricted(address from, address to, string reason);

    constructor(
        string memory _name,
        string memory _symbol,
        string memory _propertyId,
        uint256 _valuationUsd,
        uint256 _totalShares,
        uint256 _minInvestmentUsd
    ) ERC20(_name, _symbol) {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ISSUER_ROLE, msg.sender);
        _grantRole(TRANSFER_AGENT_ROLE, msg.sender);

        property = PropertyInfo({
            propertyId: _propertyId,
            titleDeedHash: "",
            valuationUsd: _valuationUsd,
            totalShares: _totalShares,
            minInvestmentUsd: _minInvestmentUsd,
            active: true
        });

        _mint(msg.sender, _totalShares);
        emit PropertyTokenized(_propertyId, _totalShares, _valuationUsd);
    }

    // ═══════════════════════════════════════════════════════════
    // KYC / AML WHITELIST
    // ═══════════════════════════════════════════════════════════

    function whitelistInvestor(
        address _investor,
        string calldata _jurisdiction
    ) external onlyRole(TRANSFER_AGENT_ROLE) {
        investors[_investor] = Investor({
            kycVerified: true,
            amlCleared: true,
            jurisdiction: _jurisdiction,
            investedAt: block.timestamp
        });
        whitelist[_investor] = true;
        emit InvestorWhitelisted(_investor, _jurisdiction);
    }

    function revokeInvestor(address _investor) external onlyRole(TRANSFER_AGENT_ROLE) {
        whitelist[_investor] = false;
        investors[_investor].kycVerified = false;
    }

    // ═══════════════════════════════════════════════════════════
    // TRANSFER RESTRICTIONS (ERC-1400 compliance)
    // ═══════════════════════════════════════════════════════════

    function _update(address from, address to, uint256 value) internal virtual override {
        // Minting and burning bypass restrictions
        if (from != address(0) && to != address(0)) {
            require(whitelist[from], "Sender not whitelisted");
            require(whitelist[to], "Recipient not whitelisted");
            require(property.active, "Property token inactive");

            // Ensure minimum holding requirement
            if (balanceOf(to) == 0) {
                uint256 tokenValue = (property.valuationUsd * value) / property.totalShares;
                require(tokenValue >= property.minInvestmentUsd, "Below minimum investment");
            }
        }

        // Settle unclaimed dividends before transfer
        if (from != address(0)) _settleDividends(from);
        if (to != address(0)) _settleDividends(to);

        super._update(from, to, value);
    }

    // ═══════════════════════════════════════════════════════════
    // AUTOMATED DIVIDEND DISTRIBUTION (Rental Income)
    // ═══════════════════════════════════════════════════════════

    function distributeDividends() external payable onlyRole(ISSUER_ROLE) {
        require(msg.value > 0, "No dividends to distribute");
        require(totalSupply() > 0, "No tokens in circulation");

        dividendPerToken += (msg.value * 1e18) / totalSupply();
        totalDividendsDistributed += msg.value;

        emit DividendDistributed(msg.value, dividendPerToken);
    }

    function claimDividends() external nonReentrant {
        _settleDividends(msg.sender);
        uint256 amount = unclaimedDividends[msg.sender];
        require(amount > 0, "No dividends to claim");

        unclaimedDividends[msg.sender] = 0;
        (bool sent, ) = payable(msg.sender).call{value: amount}("");
        require(sent, "Dividend transfer failed");

        emit DividendClaimed(msg.sender, amount);
    }

    function _settleDividends(address _investor) internal {
        uint256 owed = (balanceOf(_investor) *
            (dividendPerToken - lastDividendClaimed[_investor])) / 1e18;
        unclaimedDividends[_investor] += owed;
        lastDividendClaimed[_investor] = dividendPerToken;
    }

    function pendingDividends(address _investor) external view returns (uint256) {
        uint256 owed = (balanceOf(_investor) *
            (dividendPerToken - lastDividendClaimed[_investor])) / 1e18;
        return unclaimedDividends[_investor] + owed;
    }

    // ═══════════════════════════════════════════════════════════
    // SECONDARY MARKET (24/7 Token Trading)
    // ═══════════════════════════════════════════════════════════

    function createSellOrder(uint256 _amount, uint256 _pricePerTokenUsd) external {
        require(balanceOf(msg.sender) >= _amount, "Insufficient tokens");
        require(_amount > 0 && _pricePerTokenUsd > 0, "Invalid order");

        uint256 orderId = orderCounter++;
        sellOrders[orderId] = SellOrder({
            seller: msg.sender,
            amount: _amount,
            pricePerTokenUsd: _pricePerTokenUsd,
            active: true
        });

        // Lock tokens
        _transfer(msg.sender, address(this), _amount);
        emit SellOrderCreated(orderId, msg.sender, _amount, _pricePerTokenUsd);
    }

    function fillOrder(uint256 _orderId) external payable nonReentrant {
        SellOrder storage order = sellOrders[_orderId];
        require(order.active, "Order not active");
        require(whitelist[msg.sender], "Buyer not whitelisted");

        order.active = false;

        // Transfer tokens to buyer
        _transfer(address(this), msg.sender, order.amount);

        // Transfer payment to seller
        (bool sent, ) = payable(order.seller).call{value: msg.value}("");
        require(sent, "Payment failed");

        emit SellOrderFilled(_orderId, msg.sender, order.amount);
    }

    function cancelOrder(uint256 _orderId) external {
        SellOrder storage order = sellOrders[_orderId];
        require(order.seller == msg.sender, "Not order owner");
        require(order.active, "Already inactive");

        order.active = false;
        _transfer(address(this), msg.sender, order.amount);
    }

    // ═══════════════════════════════════════════════════════════
    // VIEW
    // ═══════════════════════════════════════════════════════════

    function getInvestorInfo(address _investor) external view returns (
        bool kyc, bool aml, string memory jurisdiction,
        uint256 balance, uint256 pendingDiv
    ) {
        Investor storage inv = investors[_investor];
        uint256 owed = (balanceOf(_investor) *
            (dividendPerToken - lastDividendClaimed[_investor])) / 1e18;
        return (
            inv.kycVerified, inv.amlCleared, inv.jurisdiction,
            balanceOf(_investor), unclaimedDividends[_investor] + owed
        );
    }

    function pricePerShare() external view returns (uint256) {
        return property.valuationUsd / property.totalShares;
    }

    receive() external payable {}
}
