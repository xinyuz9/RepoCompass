import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { GoalTitle } from "./GoalTitle";

afterEach(cleanup);

it("bounds a long issue title without losing its original text or treating it as HTML", () => {
  const goal = "需求".repeat(180) + "\n<script>unsafe()</script>\nSource details";
  render(<GoalTitle objective={goal} />);
  expect(Array.from(screen.getByRole("heading").textContent!)).toHaveLength(181);
  fireEvent.click(screen.getByText("Read full goal · 完整目标"));
  expect(screen.getByLabelText("Full investigation goal").textContent).toBe(goal);
  expect(document.querySelector("script")).toBeNull();
});

it("does not add a disclosure for a short single-line goal", () => {
  render(<GoalTitle objective="Trace imports" />);
  expect(screen.getByRole("heading")).toHaveTextContent("Trace imports");
  expect(screen.queryByLabelText("Full investigation goal")).not.toBeInTheDocument();
});
