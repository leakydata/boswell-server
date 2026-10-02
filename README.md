# Boswell Server

The home processing server for [Boswell Phone](https://github.com/leakydata/boswell-phone):
your own computer (with an NVIDIA GPU) does the heavy lifting — speech detection,
transcription, speaker separation, voiceprints and sound tagging — with models far larger
than a phone can run, and sends finished transcripts back to the phone.

**Status: being built.** The plan:

- **Private by design.** The phone reaches the server over [Tailscale](https://tailscale.com),
  a private network between your own devices; nothing is exposed to the internet. Each phone
  pairs once by scanning a QR code and authenticates every request with its own key.
- **The phone stays in charge.** People, names and voices live on the phone; the server returns
  transcripts in the phone's own format, with voiceprints the phone matches. When home is out
  of reach, the phone does the work itself.
- **A small terminal screen** (built with Textual): GPU, queue, paired phones, models, logs.
- **Models chosen by measurement** on real recordings, using CUDA where possible.

## License

[Apache License 2.0](LICENSE). **Credit is required:** keep the copyright notice and the
[NOTICE](NOTICE) file in anything you distribute.

Copyright 2026 Nathan Jones ([@leakydata](https://github.com/leakydata)).
