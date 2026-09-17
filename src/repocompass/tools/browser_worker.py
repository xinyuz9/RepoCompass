"""Trusted standalone worker. Only launched inside the no-network bwrap sandbox."""

import json
import resource
import sys


def main() -> None:
    for kind, value in (
        (resource.RLIMIT_CPU, 8),
        (resource.RLIMIT_NOFILE, 512),
        (resource.RLIMIT_NPROC, 256),
        (resource.RLIMIT_FSIZE, 8 * 1024 * 1024),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(kind, (value, value))
    # Keep importing optional dependencies out of the default application path.
    from playwright.sync_api import sync_playwright

    payload = json.loads(sys.stdin.buffer.read(2 * 1024 * 1024))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/chromium/chrome-headless-shell",
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--renderer-process-limit=1",
            ],
            timeout=5000,
        )
        context = browser.new_context(
            java_script_enabled=False,
            accept_downloads=False,
            service_workers="block",
        )
        try:
            delivered = False

            def route_request(route):
                nonlocal delivered
                request = route.request
                if (
                    not delivered
                    and request.url == payload["url"]
                    and request.method == "GET"
                    and request.is_navigation_request()
                ):
                    delivered = True
                    route.fulfill(
                        status=200,
                        body=payload["html"],
                        headers={
                            "Content-Type": "text/html; charset=utf-8",
                            "Content-Security-Policy": "default-src 'none'; form-action 'none'; base-uri 'none'; frame-src 'none'",
                            "X-Content-Type-Options": "nosniff",
                        },
                    )
                else:
                    route.abort()

            context.route("**/*", route_request)
            page = context.new_page()
            page.goto(payload["url"], wait_until="domcontentloaded", timeout=3000)
            # This fixed extraction script is trusted code, never supplied by a model/page.
            snapshot = page.evaluate("""() => ({
              title: document.title.slice(0, 256),
              text: (document.body?.innerText ?? '').slice(0, 8000),
              truncated: (document.body?.innerText.length ?? 0) > 8000,
              links: Array.from(document.querySelectorAll('a[href]')).slice(0, 32)
                .map(a => ({url: a.href.slice(0, 512), label: a.innerText.slice(0, 80)}))
            })""")
            print(json.dumps(snapshot, ensure_ascii=False))
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
