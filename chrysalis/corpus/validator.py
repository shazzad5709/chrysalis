from __future__ import annotations

import importlib
import inspect

from chrysalis.mrs.base import BaseMR
from chrysalis.registry.registry import RegistryLoader


class CorpusValidator:
    def __init__(self, registry_loader: RegistryLoader | None = None) -> None:
        self.registry_loader = registry_loader or RegistryLoader()
        self._mr_cache: dict[str, BaseMR] = {}

    def validate_pair(self, mr_id: str, source_input, followup_input) -> tuple[bool, str]:
        mr = self._get_mr_instance(mr_id)

        if mr_id == "CHR-GEN-005":
            verifier = getattr(mr, "verify_airtight", None)
            if verifier is None or not verifier(source_input, followup_input):
                return False, "airtight_guarantee_failed"

        dispatch = {
            "CHR-SA-001": self._validate_sa_001,
            "CHR-SA-007": self._validate_sa_007,
            "CHR-SA-008": self._validate_sa_008,
            "CHR-SA-010": self._validate_sa_010,
            "CHR-NLI-004": self._validate_nli_004,
            "CHR-NLI-005": self._validate_nli_005,
            "CHR-NLI-006": self._validate_nli_006,
            "CHR-GEN-005": self._validate_gen_005,
            "CHR-GEN-018": self._validate_gen_018,
            "CHR-GEN-019": self._validate_gen_019,
        }
        validator = dispatch.get(mr_id)
        if validator is None:
            return False, f"unsupported_mr:{mr_id}"
        return validator(mr, source_input, followup_input)

    def _get_mr_instance(self, mr_id: str) -> BaseMR:
        cached = self._mr_cache.get(mr_id)
        if cached is not None:
            return cached

        record = self.registry_loader.get_mr(mr_id)
        module = importlib.import_module(record["implementation_module"])
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseMR) and obj is not BaseMR and obj.__module__ == module.__name__:
                instance = obj()
                self._mr_cache[mr_id] = instance
                return instance

        msg = f"No BaseMR implementation found for {mr_id}"
        raise ValueError(msg)

    @staticmethod
    def _run_automated_checks(mr: BaseMR, source_input, followup_input, reason: str) -> tuple[bool, str]:
        if mr.automated_checks(source_input, followup_input):
            return True, ""
        return False, reason

    def _validate_sa_001(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "sa_001_checks_failed")

    def _validate_sa_007(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "sa_007_checks_failed")

    def _validate_sa_008(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "sa_008_checks_failed")

    def _validate_sa_010(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "sa_010_checks_failed")

    def _validate_nli_004(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "nli_004_checks_failed")

    def _validate_nli_005(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "nli_005_checks_failed")

    def _validate_nli_006(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "nli_006_checks_failed")

    def _validate_gen_005(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "gen_005_checks_failed")

    def _validate_gen_018(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "gen_018_checks_failed")

    def _validate_gen_019(self, mr: BaseMR, source_input, followup_input) -> tuple[bool, str]:
        return self._run_automated_checks(mr, source_input, followup_input, "gen_019_checks_failed")
