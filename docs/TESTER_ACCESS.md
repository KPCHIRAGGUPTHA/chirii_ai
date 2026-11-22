# Tester access for the pilot (proposal for S8 planning, not built)

Written 22 November 2026, at the owner's request, before S6 is built. Nothing here is implemented yet. It compares
three ways for about 20 testers to use Chirii and lists what each would need. The owner's leaning is (b).

## Where things stand today

- The app answers only requests whose network address is this laptop (`local_only` in `chirii/app/api/main.py`).
- Login: Argon2id passwords, 5 wrong tries lock a name for 15 minutes, sessions in a HttpOnly SameSite=Strict cookie
  (`Secure` only when the request is HTTPS), 12 hours idle / 7 days at most, held in memory (a restart logs everyone
  out).
- Changes must come from the app's own page: the `Origin` (or `Referer`) host must equal the `Host` header; a request
  with neither is allowed (browsers always send one on a `POST` from a page, other tools may not).
- One language model on the laptop CPU: about 1.6–1.7 s per answer for one person, longer when several ask at once.
- The data and both keys (`CHIRII_DATA_KEY`, `CHIRII_SECRET_KEY`) are on the laptop only.

## The three options

| | (a) In person on this laptop | (b) Cloudflare Tunnel from this laptop | (c) Cheap cloud server |
| :--- | :--- | :--- | :--- |
| How testers reach it | Sit at the laptop, or a phone on the same Wi-Fi (needs a LAN mode) | Their own phone or computer, anywhere, at a `https://…` address | Their own device, anywhere |
| Where the data is | Laptop only | Laptop only; traffic passes through Cloudflare | On the provider's server and disks |
| Who could read it in transit | Nobody outside the room | Cloudflare (it ends the HTTPS connection at its edge, then re-encrypts to the laptop) | The provider can reach the server and its disks |
| Model speed | Same as now | Same as now, shared by all testers; queues at busy times | A cheap CPU server is slower than this laptop; a GPU server is not cheap |
| Availability | Only when the owner is there | Only while the laptop is on, awake and online | Always on |
| Cost | None | Free plan of Cloudflare Tunnel (and Access); a domain on Cloudflare (roughly ₹800–1,000 a year; check current prices) or a temporary `trycloudflare.com` address (not for the pilot: changes on every start, no Access) | Roughly ₹1,500–4,000 a month for enough RAM for the 1.5B model on CPU (check current prices) |
| Realistic usage | Short supervised sessions; not daily use | Daily use over weeks, as a real assistant | Daily use |
| New security work | Small | Medium (below) | Large: everything in (b), plus server hardening, key storage on the server, backups, provider terms |
| Fits "data stays on the pilot laptop" in the consent note | Yes | Yes, with one added sentence about transit | No: the consent note changes in substance |

### Recommendation: (b), with Cloudflare Access in front

(b) gives real daily use without moving the data or the keys off the laptop, and the security changes are contained.
Put **Cloudflare Access** in front of the tunnel: only the 20 testers' email addresses can reach the page at all (a
one-time code to their email), and Chirii's own login comes after that. So an attacker on the internet never reaches
the Chirii login page. (a) is the fallback for the first session with each tester. (c) is not worth it for a
20-person pilot: more cost and risk, and the data leaves the laptop.

## Security changes needed for (b)

1. **Replace "this computer only" with an explicit mode, never by just removing it.** Behind the tunnel every
   request reaches the app from `cloudflared` on the laptop itself, so it already looks local: the current check
   would silently let the whole internet through. Instead:
   - A setting `CHIRII_ACCESS=local|tunnel` (default `local`, unchanged behaviour).
   - In `tunnel` mode the app listens on 127.0.0.1 only (never 0.0.0.0), and accepts a request only if it carries the
     Cloudflare Access token (`Cf-Access-Jwt-Assertion`), checked against the Access team's public keys and the
     application's audience tag. A request without a valid token is refused, even from 127.0.0.1.
   - `Host` must be the configured public hostname (`CHIRII_PUBLIC_HOST`); anything else is refused.
   - The owner's own use on the laptop stays on `http://127.0.0.1:8000` in local mode, or goes through the tunnel too.
2. **HTTPS and cookies.** Cloudflare gives HTTPS at the public address. The cookie gets `Secure` always in tunnel
   mode (not inferred from the request scheme, which is `http` between `cloudflared` and the app), and the
   `__Host-` prefix. Add `Strict-Transport-Security`.
3. **CSRF and origin.** Keep SameSite=Strict, and make the origin check strict in tunnel mode: every `POST`, `PUT`
   and `DELETE` must carry an `Origin` equal to `https://` + the public hostname; a missing `Origin` is refused.
   Optionally add a per-session CSRF token in a header. Keep the Content-Security-Policy and `X-Frame-Options`.
4. **Rate limiting and lockout.**
   - Lockout per name stays (5 tries, 15 minutes). Add a limit per client address, read from `CF-Connecting-IP`,
     which is trusted only in tunnel mode after the Access token check: for example 20 login attempts an hour per
     address, and a global cap on failed logins.
   - Limit asks and saves per user (for example 30 a minute) so one tester can't keep the only model busy; queue,
     with a "busy, try again in a moment" reply rather than timeouts.
   - Cloudflare's own rate-limiting rule as an outer layer.
   - Lockouts and sessions are in memory: a restart clears them. Acceptable for the pilot; noted.
5. **Accounts.** Still made by the owner with the command; no sign-up page. Each tester gets a temporary password
   by a separate channel and must change it on first login (needs a "change password" page). Add "log out
   everywhere" and an idle limit shorter for phones if wanted.
6. **Logs.** Cloudflare logs requests (addresses, paths, times), not page content beyond what it processes. Chirii's
   own logs stay content-free. Don't log the Access token or `CF-Connecting-IP` with user names longer than needed.
7. **The laptop itself.** BitLocker / Device Encryption on, a Windows account password, screen lock, `.env` readable
   only by the owner's account, `cloudflared` running as a service with its own tunnel credentials (kept out of the
   repo, like `.env`), and no other service exposed by the tunnel (one hostname → `http://127.0.0.1:8000` only).
8. **Tests before testers:** a request without the Access token is refused in tunnel mode even from 127.0.0.1; a
   wrong `Host` or `Origin` is refused; the cookie is `Secure` and `__Host-`; rate limits answer 429; local mode is
   unchanged (the existing per-user isolation tests run in both modes).

## Consent-note changes (a new version, so every tester accepts it again)

- Point 3 (where) becomes: "Where: on the pilot laptop only, in your own folder. When you use Chirii from your own
  phone or computer, your messages travel to the laptop over an encrypted connection through Cloudflare, which
  passes them on and does not store your memories. The language model runs on the laptop."
- New point: "Access is limited to invited testers: you sign in with your email (a one-time code from Cloudflare),
  then with your Chirii password. Cloudflare sees your email address and the times you connect."
- New point: "Chirii is available only while the pilot laptop is on and online; it may be unavailable at times."
- Keep: fictional data during the pilot, lost key means lost data, device encryption on (now for the testers' own
  devices too: "use a phone or computer with a screen lock"), how to delete, crisis numbers.
- `docs/PRIVACY.md` gets the same changes plus Cloudflare as a processor of connection data.

## Laptop uptime for (b)

- **Sleep:** set the power plan so the laptop never sleeps on mains power, and the lid can close without sleeping;
  keep it plugged in. Battery wear rises with being on all day; a battery limit (often 80%) in the maker's tool helps.
- **Restarts:** Windows Update restarts at night. Set active hours, and start `llama-server`, the app and
  `cloudflared` automatically at boot (a scheduled task or services). After a restart everyone logs in again
  (sessions are in memory).
- **Network:** home Wi-Fi drops and power cuts take it offline; the tunnel reconnects by itself when the network is
  back. A small UPS for the laptop and router covers short cuts.
- **Heat and noise:** answering uses the CPU fully for a second or two per question; fine, but keep it ventilated.
- **Being the server:** while it serves testers, heavy work on the laptop (GPU jobs, big builds) slows everyone's
  answers. Agree with testers on hours if needed ("available 8 am–11 pm").
- **Monitoring:** a simple health page checked by Cloudflare or a free uptime monitor, notifying the owner.

## Decision needed from the owner (before S8 starts)

1. Option (b) with Cloudflare Access (recommended), (a) only, or (c).
2. A domain on Cloudflare for the tunnel (needed for Access and a stable address).
3. Pilot hours, if the laptop won't be on all day.
