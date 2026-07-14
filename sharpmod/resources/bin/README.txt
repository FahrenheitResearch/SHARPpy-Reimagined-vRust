Rusty Weather backend binaries
==============================

rw_ingest and rw_sharpmod provide optional accelerated model acquisition and
point-sounding extraction. The Python GUI discovers this directory
automatically and falls back to Herbie when the binaries are unavailable or a
model is unsupported.

rw_ecape_analytic provides the standalone ecape-rs analytic MU ECAPE fast path.
It is validated against ecape-parcel-py; the application falls back first to
that Python reference and then to its local compatibility implementation if the
native executable is absent, times out, rejects a profile, or returns invalid
data.

Source: https://github.com/FahrenheitResearch/rusty-weather
License: MIT (see RUSTY-WEATHER-LICENSE.txt)
