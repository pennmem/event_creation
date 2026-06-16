import os
import sys

import pytest

from_test = False

this  = sys.modules[__name__]


def pytest_configure(config):
    this.from_test = True
    if config is not None:
        config.addinivalue_line(
            'markers',
            'rhino: integration test that needs real rhino data mounts; skipped when '
            'the r1 protocol index cannot be read.')

def pytest_unconfigure(config):
    this.from_test = False


# ---------------------------------------------------------------------------
# Session discovery (shared, reusable across test modules)
# ---------------------------------------------------------------------------
# ptsa / the r1 index are only available on rhino, so every import here is lazy:
# off-rhino, discover_sessions() returns [] and rhino-marked tests skip cleanly.

_DISCOVERY_CACHE = None


def _subject_code(subject, montage):
    """Reconstruct the code (``subject`` or ``subject_<montage>``) consumed by
    ``regression_tests.run_session_import``, which splits a code on '_'."""
    try:
        montage_num = int(montage)
    except (TypeError, ValueError):
        montage_num = 0
    return '{}_{}'.format(subject, montage_num) if montage_num else subject


def discover_sessions(config, per_experiment=2):
    """Return a diverse list of session-case dicts across experiments/system versions.

    Each case: ``{subject, subject_code, experiment, session, montage, system_version}``.
    Picks up to ``per_experiment`` sessions per experiment, preferring ones that cover
    a system version not yet seen for that experiment, so the suite exercises a broad
    spread of parsers/aligners (and surfaces a diversity of failure messages).

    Returns ``[]`` if the index/data can't be read (e.g. off rhino) so callers skip.
    """
    try:
        from ptsa.data.readers import JsonIndexReader
        reader = JsonIndexReader(os.path.join(config.paths.rhino_root, 'protocols', 'r1.json'))
        experiments = list(reader.experiments())
    except Exception:
        return []

    cases = []
    for experiment in sorted(experiments):
        seen_versions = set()
        picked = 0
        try:
            subjects = list(reader.subjects(experiment=experiment, import_type='build'))
        except Exception:
            try:
                subjects = list(reader.subjects(experiment=experiment))
            except Exception:
                continue

        for subject in subjects:
            if picked >= per_experiment:
                break
            try:
                sessions = list(reader.sessions(subject=subject, experiment=experiment))
            except Exception:
                continue
            for session in sessions:
                try:
                    versions = list(reader.aggregate_values(
                        'system_version', subject=subject, experiment=experiment, session=session))
                    version = float(versions[0]) if versions else None
                except Exception:
                    version = None
                # Prefer broadening version coverage once we already have one for this exp.
                if version is not None and version in seen_versions and picked >= 1:
                    continue
                try:
                    montage = reader.get_value('montage', subject=subject,
                                               experiment=experiment, session=session)
                except Exception:
                    montage = 0
                cases.append(dict(
                    subject=subject,
                    subject_code=_subject_code(subject, montage),
                    experiment=experiment,
                    session=int(session),
                    montage=montage,
                    system_version=version,
                ))
                seen_versions.add(version)
                picked += 1
                if picked >= per_experiment:
                    break
    return cases


def _cached_sessions():
    """Discover once per pytest run (collection calls this repeatedly)."""
    global _DISCOVERY_CACHE
    if _DISCOVERY_CACHE is None:
        try:
            from ..submission.configuration import config
            _DISCOVERY_CACHE = discover_sessions(config)
        except Exception:
            _DISCOVERY_CACHE = []
    return _DISCOVERY_CACHE


@pytest.fixture(scope="session")
def config():
    """The shared submission configuration singleton."""
    from ..submission.configuration import config as _config
    return _config


@pytest.fixture(scope="session")
def discovered_sessions(config):
    """The full diverse session list (reusable by any test module)."""
    return discover_sessions(config)


def pytest_generate_tests(metafunc):
    """Parametrize any test requesting ``session_case`` with one discovered session each."""
    if 'session_case' not in metafunc.fixturenames:
        return
    cases = _cached_sessions()
    if not cases:
        metafunc.parametrize(
            'session_case',
            [pytest.param(None, marks=pytest.mark.skip(reason='no rhino r1 index available'))])
        return
    ids = ['{subject}_{experiment}_{session}'.format(**c) for c in cases]
    metafunc.parametrize('session_case', cases, ids=ids)


if __name__ == '__main__':
    pytest_configure(None)
    print(from_test)
