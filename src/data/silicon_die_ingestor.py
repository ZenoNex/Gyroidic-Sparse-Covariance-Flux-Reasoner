"""
Silicon Die Image Ingestion & Automated Expired-Patent / Free-Content Filter.

Catalogs open IC die photomicrograph repositories (siliconprawn.org/map/, ZeptoBars,
Wikimedia Commons, TinyTapeout, Ken Shirriff, etc.), enforces strict automated patent
expiration (35 U.S.C. 154 utility term, pre-1995 17-year grant term, maintenance fee
lapse, and SCPA 17 U.S.C. 904 10-year + end-of-calendar-year mask work expiration)
and free-content license filtering, and projects verified die imagery into structural
blueprints by delegating to the canonical Gyroidic vision, codec, DSP, blueprint,
voxelboxter, octree, fossilizer, and graph subsystems.

Offline-safe and deterministic: no un-gated network requests or background threads.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlencode

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    Image = None
    _HAS_PIL = False


# ==============================================================================
# 1. OPEN SILICON DIE REPOSITORY CATALOG & URL / TILE / API BUILDERS
# ==============================================================================

SILICON_DIE_REPOSITORIES: Dict[str, Dict[str, Any]] = {
    "siliconprawn": {
        "name": "Silicon Prawn (siliconprawn.org/map/)",
        "base_url": "https://siliconprawn.org/map/",
        "tile_template": "https://siliconprawn.org/map/{vendor}/{chipname}/{z}/{x}/{y}.{ext}",
        "config_template": "https://siliconprawn.org/map/{vendor}/{chipname}/config.json",
        "single_layer_template": "https://siliconprawn.org/map/{vendor}/{chipname}/single/{vendor}_{chipname}_{layer}.{ext}",
        "tooling": ["prawnmap", "groupXIV"],
        "default_license": "CC-BY-4.0",
        "license_policy": "verify_config_attribution",
        "default_verdict": "ALLOW_WITH_VERIFICATION",
        "is_oshw": False,
    },
    "zeptobars": {
        "name": "ZeptoBars",
        "base_url": "https://zeptobars.com/",
        "default_license": "CC-BY-3.0",
        "license_policy": "allow_with_attribution",
        "default_verdict": "ALLOW",
        "is_oshw": False,
    },
    "wikimedia_commons": {
        "name": "Wikimedia Commons (Integrated circuit dies)",
        "base_url": "https://commons.wikimedia.org/wiki/Category:Integrated_circuit_dies",
        "api_url": "https://commons.wikimedia.org/w/api.php",
        "default_license": "PER_FILE_EXTMETADATA",
        "license_policy": "filter_extmetadata_license_short_name",
        "default_verdict": "ALLOW_WITH_VERIFICATION",
        "is_oshw": False,
    },
    "tinytapeout": {
        "name": "TinyTapeout / Open Chip Galleries",
        "base_url": "https://tinytapeout.com/",
        "github_url": "https://github.com/TinyTapeout",
        "default_license": "Apache-2.0",
        "license_policy": "oshw_permissive",
        "default_verdict": "ALLOW",
        "is_oshw": True,
    },
    "righto": {
        "name": "Ken Shirriff Blog (righto.com)",
        "base_url": "http://www.righto.com/",
        "default_license": "MIXED",
        "license_policy": "explicit_cc_by_only",
        "default_verdict": "ALLOW_IF_EXPLICIT_CC_BY",
        "is_oshw": False,
    },
    "visual6502": {
        "name": "Visual6502",
        "base_url": "http://www.visual6502.org/",
        "default_license": "CC-BY-NC-ND-3.0",
        "license_policy": "deny_nc_nd_media",
        "default_verdict": "DENY",
        "is_oshw": False,
    },
    "silicon_zoo": {
        "name": "Silicon Zoo (FSU)",
        "base_url": "https://micro.magnet.fsu.edu/",
        "default_license": "ARR",
        "license_policy": "deny_all_rights_reserved",
        "default_verdict": "DENY",
        "is_oshw": False,
    },
    "chipdb": {
        "name": "ChipDB",
        "base_url": "https://chipdb.org/",
        "default_license": "NON-COMMERCIAL",
        "license_policy": "deny_unless_explicit_cc",
        "default_verdict": "DENY",
        "is_oshw": False,
    },
}


def build_siliconprawn_urls(
    vendor: str,
    chipname: str,
    layer: str = "top_metal",
    z: int = 0,
    x: int = 0,
    y: int = 0,
    ext: str = "jpg",
) -> Dict[str, str]:
    """Builds canonical siliconprawn.org/map/ viewer, config.json, single-layer, and XYZ tile URLs."""
    v = re.sub(r"[^A-Za-z0-9_.\-]", "_", vendor.strip())
    c = re.sub(r"[^A-Za-z0-9_.\-]", "_", chipname.strip())
    l = re.sub(r"[^A-Za-z0-9_.\-]", "_", layer.strip())
    e = ext.lstrip(".").lower() or "jpg"
    base = f"https://siliconprawn.org/map/{v}/{c}/"
    return {
        "viewer_url": f"{base}index.html",
        "config_url": f"{base}config.json",
        "single_layer_url": f"{base}single/{v}_{c}_{l}.{e}",
        "tile_url": f"{base}{int(z)}/{int(x)}/{int(y)}.{e}",
    }


def build_wikimedia_commons_api_url(
    category: str = "Category:Integrated_circuit_dies",
    limit: int = 50,
) -> str:
    """Builds a Wikimedia Commons API query URL for die images with extmetadata license tags."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": str(max(1, min(int(limit), 500))),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    return f"https://commons.wikimedia.org/w/api.php?{urlencode(params)}"


def build_patent_lookup_endpoints(patent_id: str) -> Dict[str, str]:
    """Builds deterministic API endpoint URLs for USPTO ODP, PatentsView, EPO OPS, Lens, and Google Patents."""
    cleaned = re.sub(r"[\s,\-]+", "", str(patent_id or "")).upper()
    digits = re.sub(r"[^0-9]", "", cleaned)
    return {
        "uspto_odp_search": f"https://api.uspto.gov/api/v1/patent/applications/search?q=applicationMetaData.patentNumber:{digits}",
        "uspto_patentsview": f"https://search.patentsview.org/api/v1/patent?q={{\"patent_id\":\"{digits}\"}}",
        "epo_ops_inpadoc_legal": f"https://ops.epo.org/3.2/rest-services/published-data/publication/docdb/{cleaned}/legal",
        "lens_patent_search": "https://api.lens.org/patent/search",
        "google_patents_bigquery_table": "patents-public-data.patents.publications",
        "wipo_patentscope": f"https://patentscope.wipo.int/search/en/detail.jsf?docId={cleaned}",
    }


# ==============================================================================
# 2. AUTOMATED PATENT EXPIRATION & FREE-CONTENT FILTERING
# ==============================================================================

ALLOWED_FREE_LICENSES = frozenset({
    "CC0",
    "CC0-1.0",
    "PD",
    "PUBLIC-DOMAIN",
    "PUBLIC_DOMAIN",
    "PUBLIC DOMAIN",
    "CC-BY-3.0",
    "CC BY 3.0",
    "CC-BY-4.0",
    "CC BY 4.0",
    "CC-BY-SA-3.0",
    "CC BY-SA 3.0",
    "CC-BY-SA-4.0",
    "CC BY-SA 4.0",
    "APACHE-2.0",
    "APACHE 2.0",
    "MIT",
    "OSHW",
})

DENIED_LICENSE_MARKERS = (
    "NC",
    "ND",
    "ARR",
    "ALL RIGHTS RESERVED",
    "ALL_RIGHTS_RESERVED",
    "PROPRIETARY",
    "NON-COMMERCIAL",
    "NONCOMMERCIAL",
    "ACADEMIC-ONLY",
    "RESTRICTED",
    "CONFIDENTIAL",
)

URAA_CUTOFF_DATE = date(1995, 6, 8)

# Maintenance fee lapse event codes from USPTO ODP PTMNFEE2 / EPO INPADOC
LAPSE_EVENT_CODES = frozenset({
    "EXP.",
    "EXPIRED",
    "FP",
    "LAPSED",
    "LAPS",
    "M1551_UNPAID",
    "M1552_UNPAID",
    "M1553_UNPAID",
    "FEE_LAPSE",
})

_PATENT_ID_REGEX = re.compile(
    r"\b(?P<office>US|EP|WO|JP)[\s\-]?"
    r"(?P<number>(?:[A-Z]\d{2})?\d{1,3}(?:[,\s]?\d{3})+|\d{5,11})"
    r"(?:[\s\-]?(?P<kind>[A-Z]\d?))?\b",
    re.IGNORECASE,
)


def _parse_date(val: Any) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, (int, float)):
        yr = int(val)
        if 1900 <= yr <= 2100:
            return date(yr, 1, 1)
        return None
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date()
        except ValueError:
            continue
    return None


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        # Feb 29 -> Feb 28
        return d.replace(year=d.year + years, day=28)


def _add_days(d: date, days: int) -> date:
    return date.fromordinal(d.toordinal() + int(days))


@dataclass
class SiliconDieComplianceReport:
    admissible: bool
    veto_reasons: List[str] = field(default_factory=list)
    license_status: Dict[str, Any] = field(default_factory=dict)
    mask_work_status: Dict[str, Any] = field(default_factory=dict)
    patent_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    structural_honesty_passed: bool = True
    public_domain_tags: List[str] = field(default_factory=list)
    repo_id: str = "unknown"
    chip_identifier: str = "unknown_die"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "admissible": self.admissible,
            "veto_reasons": list(self.veto_reasons),
            "license_status": dict(self.license_status),
            "mask_work_status": dict(self.mask_work_status),
            "patent_evaluations": list(self.patent_evaluations),
            "structural_honesty_passed": self.structural_honesty_passed,
            "public_domain_tags": list(self.public_domain_tags),
            "repo_id": self.repo_id,
            "chip_identifier": self.chip_identifier,
        }


class SiliconDiePatentFilter:
    """
    Deterministic, offline-safe legal expiration and free-content compliance gate
    for silicon die photomicrographs and layout renders.
    """

    @staticmethod
    def parse_patent_ids(text: str) -> List[Dict[str, str]]:
        """Extracts normalized US/EP/WO/JP patent identifiers from text."""
        if not text:
            return []
        found: List[Dict[str, str]] = []
        seen = set()
        for m in _PATENT_ID_REGEX.finditer(text):
            office = m.group("office").upper()
            number = re.sub(r"[,\s]", "", m.group("number")).upper()
            kind = (m.group("kind") or "").upper()
            normalized = f"{office}{number}{kind}"
            if normalized not in seen:
                seen.add(normalized)
                found.append({
                    "patent_id": normalized,
                    "office": office,
                    "number": number,
                    "kind": kind,
                })
        return found

    @staticmethod
    def evaluate_license(
        license_tag: Optional[str],
        repo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Strict allowlist gate for free-content image/layout licenses."""
        raw = (license_tag or "").strip()
        if not raw and repo_id and repo_id in SILICON_DIE_REPOSITORIES:
            default_lic = SILICON_DIE_REPOSITORIES[repo_id].get("default_license", "")
            if default_lic not in ("PER_FILE_EXTMETADATA", "MIXED"):
                raw = default_lic

        if not raw:
            return {
                "allowed": False,
                "normalized": "MISSING",
                "reason": "missing_license_metadata",
            }

        upper = re.sub(r"\s+", " ", raw.upper()).strip()
        dashed = upper.replace(" ", "-")

        # Explicit deny check for NC/ND/ARR/proprietary tokens
        tokens = set(re.split(r"[\s\-_/]+", upper))
        for marker in DENIED_LICENSE_MARKERS:
            if " " in marker or "-" in marker or "_" in marker:
                if marker in upper or marker in dashed:
                    return {
                        "allowed": False,
                        "normalized": dashed,
                        "reason": f"denied_license_marker:{marker.lower()}",
                    }
            elif marker in tokens:
                return {
                    "allowed": False,
                    "normalized": dashed,
                    "reason": f"denied_license_marker:{marker.lower()}",
                }

        if upper in ALLOWED_FREE_LICENSES or dashed in ALLOWED_FREE_LICENSES:
            return {
                "allowed": True,
                "normalized": dashed,
                "reason": "allowlisted_free_license",
            }

        return {
            "allowed": False,
            "normalized": dashed,
            "reason": f"license_not_in_allowlist:{dashed}",
        }

    @staticmethod
    def evaluate_mask_work_expiration(
        registration_date: Any = None,
        first_commercial_exploitation_date: Any = None,
        fab_or_release_year: Optional[int] = None,
        as_of: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Semiconductor Chip Protection Act (SCPA, 17 U.S.C. 904):
        - start = min(registration_date, first_commercial_exploitation_date)
        - mask_exp = Dec 31 of (start.year + 10)
        """
        today = as_of or date.today()
        reg_d = _parse_date(registration_date)
        exp_d = _parse_date(first_commercial_exploitation_date)
        candidates = [d for d in (reg_d, exp_d) if d is not None]

        if candidates:
            start = min(candidates)
            mask_exp = date(start.year + 10, 12, 31)
            is_expired = today > mask_exp
            return {
                "is_expired": is_expired,
                "start_date": start.isoformat(),
                "expiration_date": mask_exp.isoformat(),
                "rule": "SCPA_17_USC_904",
            }

        if fab_or_release_year is not None:
            yr = int(fab_or_release_year)
            mask_exp = date(yr + 10, 12, 31)
            is_expired = today > mask_exp
            return {
                "is_expired": is_expired,
                "start_date": f"{yr}-01-01",
                "expiration_date": mask_exp.isoformat(),
                "rule": "SCPA_17_USC_904_RELEASE_YEAR",
            }

        return {
            "is_expired": False,
            "start_date": None,
            "expiration_date": None,
            "rule": "SCPA_UNVERIFIED_MISSING_DATE",
        }

    @staticmethod
    def evaluate_utility_patent_expiration(
        patent_record: Union[str, Dict[str, Any]],
        as_of: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates utility patent expiration under 35 U.S.C. 154:
        - Post-1995-06-08 filing: earliest_non_provisional_filing_date + 20y + PTA_days,
          capped by terminal_disclaimer_date.
        - Pre-1995-06-08 filing: max(grant_date + 17y, earliest_non_provisional_filing_date + 20y).
        - Maintenance fee lapse: unpaid fees at 3.5 / 7.5 / 11.5 years post-grant + 6m grace
          (hard lapse at 4.0 / 8.0 / 12.0 years), or confirmed ODP/INPADOC lapse code.
        """
        today = as_of or date.today()
        if isinstance(patent_record, str):
            rec: Dict[str, Any] = {"patent_id": patent_record}
        else:
            rec = dict(patent_record)

        patent_id = str(rec.get("patent_id") or rec.get("id") or "UNKNOWN").strip().upper()
        filing_d = _parse_date(
            rec.get("earliest_non_provisional_filing_date")
            or rec.get("filing_date")
        )
        grant_d = _parse_date(rec.get("grant_date") or rec.get("issue_date"))
        td_d = _parse_date(rec.get("terminal_disclaimer_date"))
        pta_days = int(rec.get("pta_days", 0) or 0)
        fee_lapsed = bool(rec.get("maintenance_fee_lapsed", False))
        event_code = str(rec.get("legal_status_code") or rec.get("odp_event_code") or "").strip().upper()
        unpaid_stage_years = rec.get("unpaid_maintenance_fee_year")

        # 1. Explicit maintenance fee lapse check
        if fee_lapsed or (event_code in LAPSE_EVENT_CODES):
            return {
                "patent_id": patent_id,
                "is_expired": True,
                "reason": "maintenance_fee_lapsed_or_inpadoc_expired",
                "expiration_date": rec.get("lapse_date") or today.isoformat(),
                "endpoints": build_patent_lookup_endpoints(patent_id),
            }

        if grant_d is not None and unpaid_stage_years is not None:
            # 3.5 -> 4.0y, 7.5 -> 8.0y, 11.5 -> 12.0y hard lapse after 6-month grace period
            stage = float(unpaid_stage_years)
            hard_lapse_years = int(math.ceil(stage + 0.5))
            lapse_date = _add_years(grant_d, hard_lapse_years)
            if today > lapse_date:
                return {
                    "patent_id": patent_id,
                    "is_expired": True,
                    "reason": f"maintenance_fee_grace_lapsed_{hard_lapse_years}y",
                    "expiration_date": lapse_date.isoformat(),
                    "endpoints": build_patent_lookup_endpoints(patent_id),
                }

        # 2. Statutory term computation (35 U.S.C. 154)
        exp_date: Optional[date] = None
        rule = "UNVERIFIED"
        if filing_d is not None:
            twenty_yr = _add_years(filing_d, 20)
            if filing_d < URAA_CUTOFF_DATE:
                rule = "35_USC_154_PRE_1995"
                if grant_d is not None:
                    seventeen_yr = _add_years(grant_d, 17)
                    exp_date = max(seventeen_yr, twenty_yr)
                else:
                    exp_date = twenty_yr
            else:
                rule = "35_USC_154_POST_1995"
                exp_date = _add_days(twenty_yr, pta_days)
                if td_d is not None and td_d < exp_date:
                    exp_date = td_d
                    rule = "35_USC_154_TERMINAL_DISCLAIMER"
        elif grant_d is not None and grant_d < URAA_CUTOFF_DATE:
            rule = "35_USC_154_PRE_1995_GRANT"
            exp_date = _add_years(grant_d, 17)

        if exp_date is None:
            return {
                "patent_id": patent_id,
                "is_expired": False,
                "reason": "unverified_active_or_missing_patent_dates",
                "expiration_date": None,
                "rule": rule,
                "endpoints": build_patent_lookup_endpoints(patent_id),
            }

        is_expired = today > exp_date
        return {
            "patent_id": patent_id,
            "is_expired": is_expired,
            "reason": "statutory_term_expired" if is_expired else "active_utility_patent",
            "expiration_date": exp_date.isoformat(),
            "rule": rule,
            "endpoints": build_patent_lookup_endpoints(patent_id),
        }

    def verify_admissibility(
        self,
        metadata: Dict[str, Any],
        as_of: Optional[date] = None,
        textbook_filter: Optional[Any] = None,
    ) -> SiliconDieComplianceReport:
        """
        Composite legal & structural-honesty pass rule:
        - STRICTLY vetoes active silicon die patents, active SCPA mask works,
          non-allowlisted/NC/ND/ARR licenses, missing provenance on modern chips,
          or bot-wall/restriction pages.
        - Passes if:
            (source is OSHW/TinyTapeout AND license allowlisted AND no active patents)
          OR
            (license allowlisted AND SCPA mask work expired AND all linked patents expired/lapsed
             AND [if no patent IDs: fab_or_release_year <= as_of.year - 21]).
        """
        today = as_of or date.today()
        meta = metadata or {}
        repo_id = str(meta.get("repo_id") or meta.get("repository") or "unknown").strip().lower()
        vendor = str(meta.get("vendor") or "generic").strip()
        chipname = str(meta.get("chipname") or meta.get("chip") or "unknown_die").strip()
        chip_id = re.sub(r"[^A-Za-z0-9_.\-]", "_", f"{vendor}_{chipname}").lower()

        veto_reasons: List[str] = []

        # 1. Structural honesty / bot-wall check via TextbookFilter
        desc_text = " ".join(
            str(meta.get(k, "")) for k in ("title", "description", "attribution", "notes") if meta.get(k)
        ).strip()
        structural_honesty_passed = True
        if desc_text:
            try:
                from src.data.textbook_filter import TextbookFilter
                if TextbookFilter.is_restriction_or_bot_wall(desc_text):
                    structural_honesty_passed = False
                    veto_reasons.append("restriction_or_bot_wall_detected")
                elif textbook_filter is not None:
                    q_report = textbook_filter.assess(desc_text, source=repo_id)
                    if not q_report.dimension_gates.get("structural_honesty", True):
                        structural_honesty_passed = False
                        veto_reasons.append("textbook_filter_structural_honesty_failed")
            except Exception:
                pass

        # 2. License allowlist check
        lic_eval = self.evaluate_license(meta.get("license"), repo_id=repo_id)
        if not lic_eval["allowed"]:
            veto_reasons.append(f"license_veto:{lic_eval['reason']}")

        # 3. Collect explicit + text-parsed patent records
        explicit_patents = list(meta.get("patents") or [])
        explicit_ids = set()
        patent_records: List[Dict[str, Any]] = []
        for p in explicit_patents:
            if isinstance(p, str):
                pid = re.sub(r"[\s,\-]+", "", p).upper()
                explicit_ids.add(pid)
                patent_records.append({"patent_id": pid})
            elif isinstance(p, dict):
                pid = re.sub(r"[\s,\-]+", "", str(p.get("patent_id") or p.get("id") or "")).upper()
                if pid:
                    explicit_ids.add(pid)
                patent_records.append(p)

        for parsed in self.parse_patent_ids(desc_text):
            if parsed["patent_id"] not in explicit_ids:
                explicit_ids.add(parsed["patent_id"])
                patent_records.append(parsed)

        patent_evals = [self.evaluate_utility_patent_expiration(pr, as_of=today) for pr in patent_records]
        for pe in patent_evals:
            if not pe["is_expired"]:
                veto_reasons.append(f"active_or_unverified_patent:{pe['patent_id']}:{pe['reason']}")

        # 4. OSHW / TinyTapeout fast-path (still requires allowlisted license + no active patents)
        repo_info = SILICON_DIE_REPOSITORIES.get(repo_id, {})
        is_oshw = bool(meta.get("is_oshw", False)) or bool(repo_info.get("is_oshw", False)) or repo_id == "tinytapeout"

        fab_year = meta.get("fab_or_release_year") or meta.get("release_year") or meta.get("year")
        if is_oshw:
            mask_eval = {
                "is_expired": True,
                "start_date": None,
                "expiration_date": None,
                "rule": "OSHW_ZERO_PATENT_ENCUMBRANCE",
            }
        else:
            mask_eval = self.evaluate_mask_work_expiration(
                registration_date=meta.get("mask_registration_date"),
                first_commercial_exploitation_date=meta.get("first_commercial_exploitation_date"),
                fab_or_release_year=int(fab_year) if fab_year is not None else None,
                as_of=today,
            )
            if not mask_eval["is_expired"]:
                veto_reasons.append(f"mask_work_active_or_unverified:{mask_eval['rule']}")

            # Vintage fallback when no patent IDs are provided
            if not patent_evals:
                if fab_year is None or int(fab_year) > (today.year - 21):
                    veto_reasons.append("unverified_modern_die_requires_21y_vintage_or_expired_patent_proof")

        admissible = len(veto_reasons) == 0

        # Build canonical public_domain_tags
        pd_tags: List[str] = []
        if admissible:
            pd_tags.extend([
                "public_domain",
                "structural_honesty",
                "silicon_die",
                f"silicon_die:{chip_id}",
            ])
            lic_norm = lic_eval["normalized"].lower()
            if lic_norm.startswith("cc0"):
                pd_tags.append(f"cc0:{lic_norm}")
            elif lic_norm.startswith("cc-by-sa"):
                pd_tags.append(f"cc-by-sa:{lic_norm}")
            elif lic_norm.startswith("cc-by"):
                pd_tags.append(f"cc-by:{lic_norm}")
            elif is_oshw:
                pd_tags.append(f"oshw:{lic_norm}")
            else:
                pd_tags.append(f"pd:{lic_norm}")

            if mask_eval.get("expiration_date"):
                pd_tags.append(f"mask_work_expired:{mask_eval['expiration_date'][:4]}")
            elif is_oshw:
                pd_tags.append("mask_work_expired:oshw")

            for pe in patent_evals:
                if pe["is_expired"]:
                    pd_tags.append(f"patent_expired:{pe['patent_id'].lower()}")

            # Deduplicate preserving order
            pd_tags = list(dict.fromkeys(pd_tags))

        return SiliconDieComplianceReport(
            admissible=admissible,
            veto_reasons=veto_reasons,
            license_status=lic_eval,
            mask_work_status=mask_eval,
            patent_evaluations=patent_evals,
            structural_honesty_passed=structural_honesty_passed,
            public_domain_tags=pd_tags,
            repo_id=repo_id,
            chip_identifier=chip_id,
        )


# ==============================================================================
# 3. SILICON DIE IMAGE INGESTOR -> STRUCTURAL BLUEPRINT COMPILER
# ==============================================================================

class SiliconDieImageIngestor:
    """
    Ingests verified open silicon die photomicrographs through:
      1. SiliconDiePatentFilter (strict active-patent & proprietary veto)
      2. ConformalLogPolarProjector (foveal scale/rotation-invariant unrolling)
      3. ConjugateColorTransport (OKLab whitening + metal/oxide contrast)
      4. TextureDSPEngine (cyclotomic taper + spectral periodicity)
      5. vision_utilities (96-dim BT.601 YCrCb / Russian Doll Chebyshev fingerprint)
      6. GyroidicCodec / GyroidImageProjector (CRT residues + Berry phases)
      7. ImageProcessor / StructuralEntanglementNet (interlaced 768-dim manifold)
      8. CanonicalProjector (when image file path is provided)
      9. AgentSubstrateBridge (non-memorization + public_domain_tags sanitization)
     10. Structural Blueprint compilation (AddonRoutine + SiliconDieBlueprintLayer)
    """

    def __init__(
        self,
        device: str = "cpu",
        codec: Optional[Any] = None,
        conformal_projector: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        color_transport: Optional[Any] = None,
        texture_dsp: Optional[Any] = None,
        canonical_projector: Optional[Any] = None,
        substrate_bridge: Optional[Any] = None,
        textbook_filter: Optional[Any] = None,
        fossilizer: Optional[Any] = None,
        graph_manager: Optional[Any] = None,
        lightweight: bool = False,
    ):
        self.device = device
        self.lightweight = lightweight
        self.patent_filter = SiliconDiePatentFilter()
        self.repositories = SILICON_DIE_REPOSITORIES

        self._codec = codec
        self._conformal_projector = conformal_projector
        self._image_processor = image_processor
        self._color_transport = color_transport
        self._texture_dsp = texture_dsp
        self._canonical_projector = canonical_projector
        self._substrate_bridge = substrate_bridge
        self._textbook_filter = textbook_filter
        self.fossilizer = fossilizer
        self.graph_manager = graph_manager

    # --- Lazy canonical component accessors (trust_existing_codebase_first) ---

    @property
    def conformal_projector(self):
        if self._conformal_projector is None:
            from src.codec.conformal_log_polar import ConformalLogPolarProjector
            self._conformal_projector = ConformalLogPolarProjector(resolution_r=32, resolution_theta=32)
        return self._conformal_projector

    @property
    def color_transport(self):
        if self._color_transport is None:
            from src.codec.image_extension import ConjugateColorTransport
            self._color_transport = ConjugateColorTransport()
        return self._color_transport

    @property
    def texture_dsp(self):
        if self._texture_dsp is None:
            from src.core.texture_dsp import TextureDSPEngine
            self._texture_dsp = TextureDSPEngine(device=self.device)
        return self._texture_dsp

    @property
    def image_processor(self):
        if self._image_processor is None:
            from src.codec.image_extension import ImageProcessor
            self._image_processor = ImageProcessor(device=self.device)
        return self._image_processor

    @property
    def codec(self):
        if self._codec is None:
            from src.codec.gyroidic_codec import GyroidicCodec, CodecConfig
            self._codec = GyroidicCodec(CodecConfig(K=5, n=4))
        return self._codec

    @property
    def substrate_bridge(self):
        if self._substrate_bridge is None:
            from src.core.agent_substrate_bridge import AgentSubstrateBridge
            self._substrate_bridge = AgentSubstrateBridge(device=self.device)
        return self._substrate_bridge

    @staticmethod
    def _coerce_image_tensor(
        image_input: Union[torch.Tensor, np.ndarray, str, Path, Any],
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """
        Normalizes a silicon die image (Tensor [C,H,W]/[H,W,C]/[H,W], ndarray, PIL.Image,
        or local file path) into a float32 tensor [3, H, W] in [0, 1].
        """
        file_path: Optional[str] = None
        if isinstance(image_input, (str, Path)):
            file_path = str(image_input)
            if not _HAS_PIL:
                raise RuntimeError("PIL is required to load silicon die images from disk.")
            pil_img = Image.open(file_path).convert("RGB")
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            return tensor.clamp(0.0, 1.0), file_path

        if _HAS_PIL and isinstance(image_input, Image.Image):
            arr = np.asarray(image_input.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).clamp(0.0, 1.0), None

        if isinstance(image_input, np.ndarray):
            t = torch.from_numpy(image_input.astype(np.float32))
        elif isinstance(image_input, torch.Tensor):
            t = image_input.detach().float().cpu()
        else:
            raise TypeError(f"Unsupported silicon die image input type: {type(image_input)}")

        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        elif t.ndim == 4:
            t = t[0]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] > 3:
            t = t[:3]
        if t.max().item() > 1.0:
            t = t / 255.0
        return t.clamp(0.0, 1.0), file_path

    @staticmethod
    def compute_die_fingerprint(rgb_chw: torch.Tensor) -> torch.Tensor:
        """
        Extracts a deterministic 96-dim BT.601 YCrCb Chebyshev fingerprint and
        passes it through canonical `get_russian_doll_projection`.
        """
        from src.codec.vision_utilities import (
            get_russian_doll_projection,
            project_multimodal_fingerprint_to_96,
        )

        # Downsample to 64x64 for fast, deterministic Chebyshev spectral extraction
        small = F.interpolate(
            rgb_chw.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
        ).squeeze(0)
        r, g, b = small[0], small[1], small[2]
        y_ch = 0.299 * r + 0.587 * g + 0.114 * b
        cr_ch = 0.5 * r - 0.418688 * g - 0.081312 * b
        cb_ch = -0.168736 * r - 0.331264 * g + 0.5 * b

        coeffs_dict: Dict[str, List[float]] = {}
        for name, ch in (("L", y_ch), ("Cr", cr_ch), ("Cb", cb_ch)):
            flat = ch.flatten()
            # 32-bin DCT-like cosine projection (Chebyshev modes T_0..T_31)
            n = flat.numel()
            idx = torch.linspace(0.0, math.pi, n, dtype=torch.float32)
            modes = []
            for k in range(32):
                val = float((flat * torch.cos(k * idx)).mean().item())
                modes.append(val)
            coeffs_dict[name] = modes

        fp_96 = project_multimodal_fingerprint_to_96(coeffs_dict)
        return get_russian_doll_projection(fp_96, k_image_max=32)

    def ingest_die_image(
        self,
        image_input: Union[torch.Tensor, np.ndarray, str, Path, Any],
        metadata: Dict[str, Any],
        as_of: Optional[date] = None,
        compile_blueprint: bool = True,
        fossilize: bool = False,
        register_graph: bool = False,
    ) -> Dict[str, Any]:
        """
        Full end-to-end Silicon Die Ingestion pipeline:
        1. Strictly vets patent expiration, SCPA mask work expiration, and free-content license.
        2. If vetoed, immediately returns `{"admissible": False, "vetoed": True, ...}`
           WITHOUT projecting or fossilizing any tensor data.
        3. If admissible, executes conformal log-polar, conjugate color transport,
           TextureDSP cyclotomic filtering, 96-dim Russian Doll fingerprint,
           GyroidicCodec CRT residues, ImageProcessor entanglement, and compiles
           a Structural Blueprint (`AddonRoutine` + `SiliconDieBlueprintLayer`).
        """
        compliance = self.patent_filter.verify_admissibility(
            metadata=metadata,
            as_of=as_of,
            textbook_filter=self._textbook_filter,
        )
        if not compliance.admissible:
            return {
                "admissible": False,
                "vetoed": True,
                "compliance": compliance.to_dict(),
                "blueprint": None,
                "public_domain_tags": [],
                "connected_subsystems": ["silicon_die_patent_filter"],
            }

        rgb_chw, file_path = self._coerce_image_tensor(image_input)
        connected: List[str] = ["silicon_die_patent_filter"]

        # 1. Conformal Log-Polar Foveal Unrolling
        log_polar = self.conformal_projector(rgb_chw)  # [3, res_r, res_theta]
        connected.append("conformal_log_polar")

        # 2. Conjugate Color Transport (metal/oxide contrast whitening)
        try:
            rgb_hwc = rgb_chw.permute(1, 2, 0).contiguous()
            transported_hwc = self.color_transport.transport_colors(rgb_hwc, kappa=0.15)
            connected.append("conjugate_color_transport")
        except Exception:
            transported_hwc = rgb_chw.permute(1, 2, 0).contiguous()

        # 3. TextureDSP Cyclotomic Taper & Spectral Energy
        try:
            tapered, _ = self.texture_dsp.apply_cyclotomic_taper_window(log_polar.mean(dim=0))
            spectral_energy = float(torch.norm(tapered).item())
            connected.append("texture_dsp")
        except Exception:
            spectral_energy = float(torch.norm(log_polar).item())

        # 4. 96-dim Russian Doll Chebyshev Fingerprint
        fingerprint_96 = self.compute_die_fingerprint(rgb_chw)
        connected.append("vision_utilities")

        # 5. GyroidicCodec CRT Residues & Berry Phases (affordance-gated)
        from src.core.hardware_monitor import ComputeAffordanceType
        with torch.no_grad():
            enc = self.codec.encode(
                text=f"silicon_die {compliance.chip_identifier}",
                image=log_polar,
                commutativity="image_first",
                image_fingerprint=fingerprint_96,
            )
            crt_residues = (
                enc.residue.detach().cpu()
                if getattr(enc, "residue", None) is not None
                else enc.encoded.detach().cpu()
            )
            berry_phases = enc.berry_phases.detach().cpu() if enc.berry_phases is not None else torch.zeros(5)
        connected.append("gyroidic_codec")

        # 6. ImageProcessor + StructuralEntanglementNet (interlaced flesh + gyroid bone)
        entangled_state: torch.Tensor
        try:
            if _HAS_PIL:
                pil_small = Image.fromarray(
                    (rgb_chw.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
                )
                with torch.no_grad():
                    entangled_state = self.image_processor(
                        pil_small, gyroid_residue=fingerprint_96.to(self.device)
                    ).detach().cpu()
            else:
                with torch.no_grad():
                    batch_64 = F.interpolate(
                        rgb_chw.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
                    ).to(self.device)
                    entangled_state = self.image_processor.visual_net(batch_64).detach().cpu()
            connected.append("image_processor")
            connected.append("structural_entanglement_net")
        except Exception:
            entangled_state = F.pad(fingerprint_96.unsqueeze(0), (0, max(0, 768 - 96)))[:, :768]

        # 7. CanonicalProjector (multi-scale L0/L1 edge-density projection if disk path)
        canonical_edge_density = 0.0
        if file_path is not None:
            try:
                if self._canonical_projector is None:
                    from src.data.canonical_projection import CanonicalProjector
                    self._canonical_projector = CanonicalProjector(dim=int(entangled_state.shape[-1]), device="cpu")
                c_proj = self._canonical_projector.project_image_path_to_state(
                    file_path, use_log_polar=True
                )
                canonical_edge_density = float(c_proj.get("edge_density_l0", 0.0))
                connected.append("canonical_projector")
            except Exception:
                pass

        # 8. Sparse Covariance Topology Signature (from log-polar channels)
        lp_flat = log_polar.reshape(3, -1)
        lp_centered = lp_flat - lp_flat.mean(dim=1, keepdim=True)
        cov_3x3 = (lp_centered @ lp_centered.T) / max(1, lp_centered.shape[1] - 1)

        # 9. AgentSubstrateBridge Non-Memorization & Public-Domain Tag Sanitization
        pot_crt = tuple(
            int(abs(float(crt_residues[i].mean().item())) * 100) % 97
            for i in range(min(3, crt_residues.shape[0]))
        )
        while len(pot_crt) < 3:
            pot_crt = pot_crt + (0,)

        digest = hashlib.blake2s(
            f"{compliance.chip_identifier}:{compliance.repo_id}".encode("utf-8"),
            digest_size=8,
        ).hexdigest()

        sanitized = self.substrate_bridge.sanitize_and_verify_non_memorization({
            "pot_identity_crt": pot_crt,
            "blake2s_digest": digest,
            "description": f"structural_soliton_crt_{pot_crt[0]}_{pot_crt[1]}_{pot_crt[2]}_{digest}",
            "tags": list(compliance.public_domain_tags),
            "meta_state_shielded": entangled_state.squeeze(0).clone(),
            "dyad_metadata": {
                "repo_id": compliance.repo_id,
                "chip_identifier": compliance.chip_identifier,
                "license": compliance.license_status.get("normalized", "UNKNOWN"),
                "spectral_energy": round(spectral_energy, 6),
                "compute_affordance": ComputeAffordanceType.COMPUTE_BATCH.value,
            },
        })
        connected.append("agent_substrate_bridge")

        # 10. Extract blueprint grid (8x8 metal/poly/diffusion structural heightmap from log-polar)
        grid_8x8 = F.interpolate(
            transported_hwc.permute(2, 0, 1).unsqueeze(0).mean(dim=1, keepdim=True),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        blueprint_payload: Dict[str, Any] = {
            "die_id": compliance.chip_identifier,
            "repo_id": compliance.repo_id,
            "pot_identity_crt": pot_crt,
            "public_domain_tags": list(sanitized.get("public_domain_tags", compliance.public_domain_tags)),
            "fingerprint_96": fingerprint_96,
            "crt_residues": crt_residues,
            "berry_phases": berry_phases,
            "entangled_state": entangled_state,
            "sparse_covariance_3x3": cov_3x3,
            "spectral_energy": spectral_energy,
            "canonical_edge_density": canonical_edge_density,
            "layer_grid_8x8": grid_8x8,
            "structural_algebra_attestation": sanitized.get("structural_algebra_attestation", {}),
            "compliance": compliance.to_dict(),
        }

        addon_routine = None
        structural_graph = None
        if compile_blueprint:
            from src.core.structural_blueprints import AddonRoutine
            addon_routine = AddonRoutine.compile_silicon_die_blueprint(blueprint_payload)
            structural_graph = addon_routine.generate_graph()
            connected.append("structural_blueprints")

        # 11. Optional Fossilization via KnowledgeDyadFossilizer
        fossil_path = None
        if fossilize and self.fossilizer is not None:
            try:
                from src.core.knowledge_dyad_fossilizer import KnowledgeDyad
                feat_dim = int(getattr(self.fossilizer, "feature_dim", 512))
                raw_state = entangled_state.squeeze(0).detach().cpu()
                if raw_state.numel() < feat_dim:
                    seed_vec = F.pad(raw_state, (0, feat_dim - raw_state.numel()))
                else:
                    seed_vec = raw_state[:feat_dim]
                dyad = KnowledgeDyad(
                    image_fingerprint=fingerprint_96,
                    linguistic_description=sanitized["description"],
                    relevance_score=1.0,
                    metadata={
                        "source": compliance.repo_id,
                        "chip_identifier": compliance.chip_identifier,
                        "tags": sanitized.get("tags", []),
                        "public_domain_tags": blueprint_payload["public_domain_tags"],
                        "silicon_die_blueprint": True,
                        "structural_honesty": True,
                    },
                )
                fossil_path = self.fossilizer.fossilize(
                    dyad,
                    seed_vec,
                    seed_vec,
                )
                connected.append("knowledge_dyad_fossilizer")
            except Exception:
                fossil_path = None

        # 12. Optional Graph Registration via GyroidicGraphManager
        graph_node = None
        if register_graph and self.graph_manager is not None:
            try:
                from src.topology.embedding_graph import KnowledgeFossilNode
                gm_dim = int(getattr(self.graph_manager, "dim", 64))
                raw_state = entangled_state.squeeze(0).detach().cpu()
                if raw_state.numel() < gm_dim:
                    node_state = F.pad(raw_state, (0, gm_dim - raw_state.numel()))
                else:
                    node_state = raw_state[:gm_dim]
                node = KnowledgeFossilNode(
                    node_id=f"die_{compliance.chip_identifier}_{digest[:6]}",
                    state=node_state,
                    text=sanitized["description"],
                    metrics={
                        "tags": sanitized.get("tags", []),
                        "public_domain_tags": blueprint_payload["public_domain_tags"],
                        "repo_id": compliance.repo_id,
                        "silicon_die_blueprint": True,
                        "pot_identity_crt": pot_crt,
                    },
                )
                if hasattr(self.graph_manager, "nodes"):
                    self.graph_manager.nodes.append(node)
                graph_node = node
                connected.append("embedding_graph")
            except Exception:
                graph_node = None

        return {
            "admissible": True,
            "vetoed": False,
            "compliance": compliance.to_dict(),
            "blueprint": blueprint_payload,
            "addon_routine": addon_routine,
            "structural_graph": structural_graph,
            "public_domain_tags": blueprint_payload["public_domain_tags"],
            "fossil_path": fossil_path,
            "graph_node": graph_node,
            "connected_subsystems": connected,
        }
