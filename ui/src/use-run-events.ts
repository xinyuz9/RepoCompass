import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { observeRunEvents, type StreamState } from "./event-stream";

export function useRunEvents(runId: string | null, threadId: string | null, enabled: boolean, refreshEpoch: number) {
  const client = useQueryClient();
  const [state, setState] = useState<StreamState>("connecting");
  useEffect(() => {
    if (!enabled || !runId || !threadId || typeof EventSource === "undefined") return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const changed = () => {
      if (timer !== undefined) return;
      timer = setTimeout(() => {
        timer = undefined;
        void client.invalidateQueries({ queryKey: ["dashboard", "runs", runId] });
        void client.invalidateQueries({ queryKey: ["dashboard", "threads", threadId] });
      }, 300);
    };
    const subscription = observeRunEvents({ runId, create: (url) => new EventSource(url), changed, state: setState });
    const visibility = () => document.hidden ? subscription.pause() : subscription.resume();
    visibility();
    document.addEventListener("visibilitychange", visibility);
    return () => {
      subscription.close();
      if (timer !== undefined) clearTimeout(timer);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [client, enabled, runId, threadId, refreshEpoch]);
  return state;
}
