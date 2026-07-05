from src.ingest.fetch_disruptions import parse_situations

SIRI = """<?xml version="1.0" encoding="UTF-8"?>
<Siri xmlns="http://www.siri.org.uk/siri" version="2.0">
  <ServiceDelivery>
    <SituationExchangeDelivery>
      <Situations>
        <PtSituationElement>
          <SituationNumber>abc-1</SituationNumber>
          <ParticipantRef>Leeds</ParticipantRef>
          <Progress>open</Progress>
          <MiscellaneousReason>roadworks</MiscellaneousReason>
          <Planned>true</Planned>
          <Summary>Closure on High St</Summary>
          <ValidityPeriod><StartTime>2026-07-01T00:00:00Z</StartTime><EndTime>2026-07-09T00:00:00Z</EndTime></ValidityPeriod>
          <Consequences><Consequence><Affects><StopPoints>
            <AffectedStopPoint><StopPointRef>450012345</StopPointRef></AffectedStopPoint>
            <AffectedStopPoint><StopPointRef>450099999</StopPointRef></AffectedStopPoint>
          </StopPoints></Affects></Consequence></Consequences>
        </PtSituationElement>
        <PtSituationElement>
          <SituationNumber>abc-2</SituationNumber>
          <Progress>open</Progress>
          <MiscellaneousReason>accident</MiscellaneousReason>
          <Planned>false</Planned>
          <Summary>Bristol incident</Summary>
          <Consequences><Consequence><Affects><StopPoints>
            <AffectedStopPoint><StopPointRef>010012345</StopPointRef></AffectedStopPoint>
          </StopPoints></Affects></Consequence></Consequences>
        </PtSituationElement>
        <PtSituationElement>
          <SituationNumber>abc-3</SituationNumber>
          <Progress>open</Progress>
          <MiscellaneousReason>weather</MiscellaneousReason>
          <Planned>false</Planned>
          <Summary>Region-wide warning, no stops named</Summary>
        </PtSituationElement>
      </Situations>
    </SituationExchangeDelivery>
  </ServiceDelivery>
</Siri>
"""


def test_wy_situation_kept_with_stop_refs():
    df = parse_situations(SIRI.encode(), "2026-07-02T12:00:00")
    row = df[df.situation_id == "abc-1"].iloc[0]
    assert row.wy_specific
    assert row.wy_stop_refs == "450012345,450099999"
    assert row.reason == "roadworks"
    assert bool(row.planned) is True
    assert row.validity_start == "2026-07-01T00:00:00Z"


def test_non_wy_situation_dropped():
    df = parse_situations(SIRI.encode(), "2026-07-02T12:00:00")
    assert "abc-2" not in set(df.situation_id)  # Bristol stop only -> not WY


def test_blanket_situation_kept_but_not_wy_specific():
    df = parse_situations(SIRI.encode(), "2026-07-02T12:00:00")
    row = df[df.situation_id == "abc-3"].iloc[0]
    assert not row.wy_specific  # names no stops: kept, flagged
    assert row.n_affected_stops == 0
