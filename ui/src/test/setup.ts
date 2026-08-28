import "@testing-library/jest-dom/vitest";

class ResizeObserverMock implements ResizeObserver {
  disconnect(): void {}
  observe(): void {}
  unobserve(): void {}
}

globalThis.ResizeObserver = ResizeObserverMock;
