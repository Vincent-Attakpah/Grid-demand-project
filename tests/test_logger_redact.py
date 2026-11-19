from nowcaster.logger.run import redact


def test_redact_url_key(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    msg = "403 for url: https://api.eia.gov/v2/x/?api_key=ABC123%0A&frequency=hourly"
    out = redact(msg)
    assert "ABC123" not in out and "api_key=***&frequency" in out


def test_redact_bare_key(monkeypatch):
    monkeypatch.setenv("EIA_API_KEY", "SECRETKEY\n")
    assert "SECRETKEY" not in redact("bad key SECRETKEY in text")
