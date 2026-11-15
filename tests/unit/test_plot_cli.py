"""Campaign plotting keeps each profile's output separate."""

import pytest

from jaxrans import plot


@pytest.mark.parametrize("custom_output", [False, True])
def test_campaign_outputs_do_not_overwrite_other_machs(tmp_path, monkeypatch, custom_output):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    profiles = [campaign / f"mach_{mach}" for mach in ("0.1", "0.2")]
    for profile in profiles:
        profile.mkdir()
        (profile / "polar.dat").touch()
    (campaign / "figures").mkdir()
    calls = []
    monkeypatch.setattr(plot, "plot_polar", lambda directory, out: calls.append((directory, out)))
    monkeypatch.setattr(plot, "plot_cp_family", lambda directory, out: calls.append((directory, out)))
    output = tmp_path / "plots"
    args = [str(campaign)] + (["--out", str(output)] if custom_output else [])
    assert plot.main(args) == 0
    for profile in profiles:
        target = output / profile.name if custom_output else profile / "figures"
        assert (profile, target / "polar.png") in calls
        assert (profile, target / "cp_family.png") in calls
    assert len(calls) == 4


def test_empty_campaign_reports_cli_error(tmp_path):
    with pytest.raises(SystemExit) as error:
        plot.main([str(tmp_path)])
    assert error.value.code == 2
