const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PropOSEscrow - DLD direct payment", function () {
  it("releases funds to payoutAddress when title is verified", async function () {
    const [buyer, seller, settlementAgent, payoutAddress] = await ethers.getSigners();

    const Escrow = await ethers.getContractFactory("PropOSEscrow");
    const escrow = await Escrow.deploy();
    await escrow.waitForDeployment();

    await escrow.connect(buyer).createDeal(seller.address, settlementAgent.address, "DLD-001");
    const dealId = 0;

    const totalAmount = ethers.parseEther("1.0");
    await escrow.connect(buyer).fundDeal(dealId, { value: totalAmount });

    await escrow.connect(settlementAgent).verifyTitle(dealId, "hash-123", payoutAddress.address);
    await escrow.connect(settlementAgent).passInspection(dealId);
    await escrow.connect(settlementAgent).verifyZKP(dealId);

    await escrow.connect(buyer).submitSignature(dealId);

    const balBefore = await ethers.provider.getBalance(payoutAddress.address);
    await escrow.connect(settlementAgent).submitSignature(dealId);
    const balAfter = await ethers.provider.getBalance(payoutAddress.address);

    const releaseAmount = totalAmount - (totalAmount * 500n / 10000n);
    expect(balAfter - balBefore).to.equal(releaseAmount);

    const status = await escrow.getDealStatus(dealId);
    expect(status[0]).to.equal(6); // Released
    expect(status[1]).to.equal(payoutAddress.address);
  });
});
