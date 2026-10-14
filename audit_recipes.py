"""Audit true cleanliness of the recipes DataFrame.

Checks completeness, category consistency, string-length validity,
and conceptual uniqueness. Uses pandas only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clean_recipes import (
    TARGET_COLUMNS,
    TEXT_COLUMNS,
    clean_text,
    enforce_id_integrity,
    handle_missing,
    load_recipes,
)

PLACEHOLDERS = ("n/a", "na", "none", "unknown", "null", "nan")
MIN_CHARS = 3
DEFAULT_CSV_PATH = Path("recipes.csv")


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def check_completeness(recipes: pd.DataFrame) -> pd.DataFrame:
    """Count hidden missing values in text columns."""
    print("\n" + "=" * 72)
    print("1. COMPLETENESS — hidden missing values")
    print("=" * 72)

    rows = []
    for col in TEXT_COLUMNS:
        text = recipes[col].fillna("").astype(str)
        stripped = text.str.strip()
        lower = stripped.str.lower()

        empty = text.eq("")
        whitespace_only = text.str.fullmatch(r"\s+", na=False)
        placeholder = lower.isin(PLACEHOLDERS)
        hidden = empty | whitespace_only | placeholder

        counts = {
            "column": col,
            "empty_string": int(empty.sum()),
            "whitespace_only": int(whitespace_only.sum()),
            "placeholder": int(placeholder.sum()),
            "hidden_missing_total": int(hidden.sum()),
        }
        rows.append(counts)

        print(f"\n{col}")
        print(f"  empty strings ('')           : {counts['empty_string']:,}")
        print(f"  whitespace-only              : {counts['whitespace_only']:,}")
        print(f"  placeholders {PLACEHOLDERS}: {counts['placeholder']:,}")
        print(f"  hidden missing total         : {counts['hidden_missing_total']:,}")

        if counts["placeholder"]:
            print("  placeholder breakdown:")
            print(
                lower[placeholder]
                .value_counts()
                .rename_axis("token")
                .to_string()
                .replace("\n", "\n    ")
            )

    summary = pd.DataFrame(rows).set_index("column")
    print("\nCompleteness summary")
    print(summary.to_string())
    return summary


def check_consistency(recipes: pd.DataFrame) -> pd.Series:
    """Sorted unique RecipeCategory values with frequencies."""
    print("\n" + "=" * 72)
    print("2. CONSISTENCY — RecipeCategory unique values and frequencies")
    print("=" * 72)

    categories = (
        recipes["RecipeCategory"]
        .fillna("Unknown")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
    )
    print(f"\nUnique categories: {len(categories):,}")
    print("Sorted alphabetically to surface near-duplicates (Dessert / Desserts):\n")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(categories.rename("frequency").to_string())
    return categories


def check_validity(recipes: pd.DataFrame, min_chars: int = MIN_CHARS) -> pd.DataFrame:
    """String-length stats and rows below a logical character threshold."""
    print("\n" + "=" * 72)
    print(f"3. VALIDITY — string lengths (flag < {min_chars} characters)")
    print("=" * 72)

    flagged_parts = []
    for col in ("Name", "RecipeIngredientParts"):
        lengths = recipes[col].fillna("").astype(str).str.len()
        print(f"\n{col} length")
        print(f"  min : {int(lengths.min())}")
        print(f"  max : {int(lengths.max())}")
        print(f"  mean: {lengths.mean():.2f}")

        short_mask = lengths < min_chars
        short_n = int(short_mask.sum())
        print(f"  rows < {min_chars} chars: {short_n:,}")
        if short_n:
            sample = recipes.loc[short_mask, ["RecipeId", col]].assign(**{f"{col}_len": lengths[short_mask]})
            flagged_parts.append(sample)
            print("  flagged rows:")
            with pd.option_context("display.max_rows", 50, "display.max_colwidth", 80, "display.width", 160):
                print(sample.head(50).to_string(index=False))
            if short_n > 50:
                print(f"  ... {short_n - 50:,} additional flagged rows not shown")

    if not flagged_parts:
        print("\nNo rows fell below the length threshold.")
        return pd.DataFrame(columns=["RecipeId"])

    flagged = flagged_parts[0]
    for extra in flagged_parts[1:]:
        flagged = flagged.merge(extra, on="RecipeId", how="outer")
    return flagged


def check_uniqueness(recipes: pd.DataFrame) -> pd.DataFrame:
    """Conceptual duplicates: identical Name + RecipeIngredientParts, ignoring RecipeId."""
    print("\n" + "=" * 72)
    print("4. UNIQUENESS — conceptual duplicates (Name + RecipeIngredientParts)")
    print("=" * 72)

    keys = pd.DataFrame(
        {
            "Name": _normalized_text(recipes["Name"]),
            "RecipeIngredientParts": _normalized_text(recipes["RecipeIngredientParts"]),
        }
    )
    duplicate_mask = keys.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    extra_copies = int(keys.duplicated(keep="first").sum())
    unique_combos = int(keys.drop_duplicates().shape[0])
    duplicate_groups = (
        keys[duplicate_mask]
        .groupby(["Name", "RecipeIngredientParts"], sort=False)
        .size()
        .reset_index(name="occurrences")
        .sort_values(["occurrences", "Name"], ascending=[False, True])
    )

    print(f"\nTotal rows                         : {len(recipes):,}")
    print(f"Unique Name + ingredients combos   : {unique_combos:,}")
    print(f"Duplicate groups (combo appears 2+): {len(duplicate_groups):,}")
    print(f"Rows involved in those groups      : {duplicate_rows:,}")
    print(f"Extra copies beyond the first      : {extra_copies:,}")

    if len(duplicate_groups):
        print("\nTop duplicate groups:")
        with pd.option_context("display.max_colwidth", 70, "display.width", 160):
            print(duplicate_groups.head(20).to_string(index=False))
    else:
        print("\nNo conceptual duplicates found.")

    return duplicate_groups


def audit_recipes(recipes: pd.DataFrame, min_chars: int = MIN_CHARS) -> dict:
    """Run all four data-quality checks on a recipes DataFrame."""
    missing = [col for col in TARGET_COLUMNS if col not in recipes.columns]
    if missing:
        raise ValueError(f"recipes is missing required columns: {missing}")

    print("Recipe data-quality audit")
    print(f"Rows: {len(recipes):,} | Columns: {list(recipes.columns)}")

    completeness = check_completeness(recipes)
    categories = check_consistency(recipes)
    flagged = check_validity(recipes, min_chars=min_chars)
    duplicates = check_uniqueness(recipes)
    return {
        "completeness": completeness,
        "categories": categories,
        "flagged_short_rows": flagged,
        "duplicate_groups": duplicates,
    }


if __name__ == "__main__":
    recipes = load_recipes(DEFAULT_CSV_PATH)
    recipes = handle_missing(recipes)
    recipes = clean_text(recipes)
    recipes = enforce_id_integrity(recipes)
    audit_recipes(recipes)
