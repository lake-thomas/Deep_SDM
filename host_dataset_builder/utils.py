from .common import *

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def geometry_union(geometries):
    """Return a union geometry across GeoPandas versions."""
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union

def slugify_species_name(name: str) -> str:
    """Convert species name to a consistent filesystem-safe slug."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

def display_name_from_slug(slug: str) -> str:
    """Convert a slug to a title-style label while preserving underscores."""
    return "_".join([x.capitalize() for x in slug.split("_") if x])

def infer_species_slug_from_filename(csv_path: Path) -> str:
    """Infer a species slug from common occurrence CSV filename patterns."""
    stem = csv_path.stem
    stem = re.sub(r"_thinned$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_thin.*$", "", stem, flags=re.IGNORECASE)
    return slugify_species_name(stem)

