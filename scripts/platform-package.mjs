import { readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [platform, dir] = process.argv.slice(2);
const [os, cpu] = platform.split("-");
const { name, license, author, repository, optionalDependencies } = (await import("../package.json", { with: { type: "json" } })).default;
writeFileSync(
  join(dir, "package.json"),
  JSON.stringify(
    {
      name: `${name}-${platform}`,
      version: optionalDependencies[`${name}-${platform}`],
      description: `The native module and the models of ${name} for ${platform}`,
      license,
      author,
      repository,
      os: [os],
      cpu: [cpu],
      ...(os === "linux" ? { libc: ["glibc"] } : {}),
      files: readdirSync(dir).filter((f) => f !== "package.json"),
    },
    null,
    2,
  ) + "\n",
);
