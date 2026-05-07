const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);
  console.log("Balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)));

  // Deploy Escrow
  const Escrow = await ethers.getContractFactory("PropOSEscrow");
  const escrow = await Escrow.deploy();
  await escrow.waitForDeployment();
  console.log("PropOSEscrow:", await escrow.getAddress());

  // Deploy Security Token (example: Palm Jumeirah Villa)
  const Token = await ethers.getContractFactory("PropOSSecurityToken");
  const token = await Token.deploy(
    "PropOS Palm Villa Shares",   // name
    "PALM-001",                   // symbol
    "DXB-PALM-4BR-001",          // propertyId
    3500000,                      // valuationUsd ($3.5M)
    10000,                        // totalShares
    545                           // minInvestmentUsd (~AED 2000)
  );
  await token.waitForDeployment();
  console.log("PropOSSecurityToken:", await token.getAddress());

  // Verify on Etherscan
  if (network.name !== "localhost") {
    console.log("Waiting for confirmations...");
    await escrow.deploymentTransaction().wait(5);
    await token.deploymentTransaction().wait(5);
    
    await hre.run("verify:verify", { address: await escrow.getAddress(), constructorArguments: [] });
    console.log("Contracts verified");
  }
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
