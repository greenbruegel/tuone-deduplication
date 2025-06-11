""" 
Basic text normalisation
1[x] Convert text to lower-case -> to_lower
2[x] Apply Unicode NFKD normalisation with pyunormalize -> normalize_nfkd
3[x] Strip leading/trailing whitespace with lstrip and rstrip -> to_strip
4[x] Remove or standardise ordinary punctuation that is not meaningful in names (“.”, “,”, “;”) -> remove_punctuation
5[x] Remove diacritics/accents (e.g. “São Paulo” → “Sao Paulo”) -> remove_diacritics
6[x] Standardise common abbreviations & symbols (& → and, % → percent) -> expand_symbols
7[x] Replace hyphens with spaces -> replace_hyphen_with_space

"""

import pandas as pd
import unicodedata
import re
from typing import Callable, Dict, List, Tuple


symbol_mapping: Dict[str, str] = {
    "&": "and",
    "＆": "and",
    "%": "percent",
    "％": "percent",
}

class TextCleaner:
    def __init__(self):
        # Precompile any shared regexes once
        self._punctuation_pattern = re.compile(r"[.,;!?]")
        self._symbol_pattern = re.compile(
            r"({})".format("|".join(map(re.escape, symbol_mapping.keys()))),
            flags=re.UNICODE,
        )

    def _generic_apply(
        self,
        df: pd.DataFrame,
        subset_cols: List[str],
        transform_str: Callable[[str], str],
        transform_list: Callable[[str], str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        For each col in subset_cols, apply transform_str to all string cells and transform_list to list cells.
        Returns (modified_df, change_counts).
        """
        change_counts: Dict[str, int] = {col: 0 for col in subset_cols}

        for col in subset_cols:
            s = df[col]
            mask_list = s.apply(lambda x: isinstance(x, list))

            # 1) Process all string cells at once
            old_str = s[~mask_list].astype("string")  # coercion if necessary
            if not old_str.empty:
                # If transform_str is one of Python's built-in string methods (lower, strip, etc.),
                # then old_str.str.lower() is faster than old_str.map(lambda t: transform_str(t)).
                # So we check if transform_str is exactly x.lower or x.strip:
                if transform_str is str.lower:
                    new_str = old_str.str.lower()
                elif transform_str is str.strip:
                    new_str = old_str.str.strip()
                elif transform_str is str.casefold:
                    new_str = old_str.str.casefold()
                else:
                    # e.g. remove_diacritics
                    new_str = old_str.map(transform_str)
                changed_str = (old_str != new_str).sum()
                df.loc[~mask_list, col] = new_str
            else:
                changed_str = 0

            # 2) Process list cells (if any)
            cnt_list_changed = 0
            if mask_list.any() and transform_list is not None:
                def _transform_list_cell(lst):
                    new_lst = []
                    cnt = 0
                    for item in lst:
                        if isinstance(item, str):
                            new_item = transform_list(item)
                            if new_item != item:
                                cnt += 1
                            new_lst.append(new_item)
                        else:
                            new_lst.append(item)
                    return new_lst, cnt

                tuples = s[mask_list].apply(_transform_list_cell)
                df.loc[mask_list, col] = tuples.apply(lambda t: t[0])
                cnt_list_changed = tuples.apply(lambda t: t[1]).sum()

            change_counts[col] = int(changed_str + cnt_list_changed)

        return df, change_counts

    def to_lower(self, df: pd.DataFrame, subset_cols: List[str]):
        def _to_lower(s: str) -> str:
            if not isinstance(s, str):
                return s
            return s.lower()
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_to_lower,
            transform_list=_to_lower
        )

    def remove_diacritics(self, df: pd.DataFrame, subset_cols: List[str]):
        def _strip_accents(s: str) -> str:
            if not isinstance(s, str):
                return s
            return "".join(
                c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn"
            )
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_strip_accents,
            transform_list=_strip_accents
        )

    def normalize_nfkd(self, df: pd.DataFrame, subset_cols: List[str]):
        def _norm_nfkd(s: str) -> str:
            if not isinstance(s, str):
                return s
            return unicodedata.normalize("NFKD", s)
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_norm_nfkd,
            transform_list=_norm_nfkd
        )

    def strip(self, df: pd.DataFrame, subset_cols: List[str]):
        def _strip(s: str) -> str:
            if not isinstance(s, str):
                return s
            return s.strip()
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_strip,
            transform_list=_strip
        )

    def remove_punctuation(self, df: pd.DataFrame, subset_cols: List[str]):
        def _strip_punct(s: str) -> str:
            if not isinstance(s, str):
                return s
            return self._punctuation_pattern.sub("", s)
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_strip_punct,
            transform_list=_strip_punct
        )

    def replace_hyphen_with_space(self, df: pd.DataFrame, subset_cols: List[str]):
        def _hyphen_space(s: str) -> str:
            if not isinstance(s, str):
                return s
            return s.replace("-", " ")
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_hyphen_space,
            transform_list=_hyphen_space
        )

    def expand_symbols(self, df: pd.DataFrame, subset_cols: List[str]):
        def _sym_expand(s: str) -> str:
            if not isinstance(s, str):
                return s
            return self._symbol_pattern.sub(
                lambda m: symbol_mapping[m.group(0)],
                s
            )
        return self._generic_apply(
            df=df,
            subset_cols=subset_cols,
            transform_str=_sym_expand,
            transform_list=_sym_expand
        )
