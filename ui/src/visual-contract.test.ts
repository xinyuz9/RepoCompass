import { describe, expect, it } from "vitest";

import mutationsCss from "./mutations.css?raw";
import observabilityCss from "./observability.css?raw";
import stylesCss from "./styles.css?raw";

const stylesheets = [stylesCss, observabilityCss, mutationsCss];

describe("portfolio visual contract", () => {
  it("does not use unreadable sub-9px typography", () => {
    const sizes = stylesheets.flatMap((stylesheet) => [
      ...[...stylesheet.matchAll(/font-size:\s*(\d+)px/g)].map((match) => Number(match[1])),
      ...[...stylesheet.matchAll(/font:\s*(?:normal\s+)?(\d+)px/g)].map((match) => Number(match[1])),
    ]);

    expect(sizes.length).toBeGreaterThan(0);
    expect(Math.min(...sizes)).toBeGreaterThanOrEqual(9);
  });

  it("keeps secondary text above WCAG AA contrast on the panel surface", () => {
    const panel = cssVariable(stylesCss, "panel");
    const mutedSecondary = cssVariable(stylesCss, "muted-2");

    expect(contrastRatio(panel, mutedSecondary)).toBeGreaterThanOrEqual(4.5);
  });

  it("uses fluid mobile lens navigation instead of fixed overflowing widths", () => {
    expect(stylesCss).toContain(".lens-rail nav { width: 100%; flex-direction: row; gap: 0; }");
    expect(stylesCss).toContain("width: auto; min-width: 0; height: 56px; flex: 1;");
  });
});

function cssVariable(stylesheet: string, name: string): string {
  const match = stylesheet.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!match?.[1]) throw new Error(`Missing CSS variable --${name}`);
  return match[1];
}

function contrastRatio(first: string, second: string): number {
  const [lighter, darker] = [relativeLuminance(first), relativeLuminance(second)].sort(
    (left, right) => right - left,
  );
  if (lighter === undefined || darker === undefined) throw new Error("Invalid color");
  return (lighter + 0.05) / (darker + 0.05);
}

function relativeLuminance(hex: string): number {
  const channels = [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255);
  const linear = channels.map((channel) =>
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
  );
  return 0.2126 * (linear[0] ?? 0) + 0.7152 * (linear[1] ?? 0) + 0.0722 * (linear[2] ?? 0);
}
