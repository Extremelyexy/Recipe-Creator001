# Recipe Dataset Pipeline

This project cleans, audits, and vectorizes a **500k+ row Food.com recipes dataset** for collaborative data science work. The pipeline loads four core fields (`RecipeId`, `Name`, `RecipeCategory`, `RecipeIngredientParts`), applies pandas-based cleaning and data-quality checks, and consolidates fragmented categories so the table is ready for downstream modeling and text vectorization.

The raw file (`recipes.csv`) is larger than GitHub’s 100 MB limit, so it is **not** stored in this repository. Each collaborator downloads it separately and generates the cleaned dataset locally.

## Setup & Installation

1. Clone the repository:

```bash
git clone https://github.com/smn1865/Recipe-Creator.git
cd Recipe-Creator
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

3. Download the raw `recipes.csv` file from the shared cloud drive:

**[INSERT CLOUD DRIVE LINK HERE]**

4. Place `recipes.csv` in the **repository root** (the same directory as `clean_recipes.py` and `audit_recipes.py`).

## Usage

Run these commands from the repository root after `recipes.csv` is in place.

Audit data quality (hidden missing values, category consistency, short strings, conceptual duplicates):

```bash
python3 audit_recipes.py
```

Load, clean, and inspect the dataset (including the top-category chart):

```bash
python3 clean_recipes.py
```

Both scripts read `recipes.csv` from the project root. They do not overwrite the raw file; each teammate produces the cleaned view locally on their own machine.
