"""Pydantic-settings config, shared by all acquisition modules."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment or defaults.

    All values overridable via env vars with prefix ``BERLIN_LST_``,
    e.g. ``BERLIN_LST_BERLIN_BBOX``.
    """

    berlin_bbox: tuple[float, float, float, float] = (
        13.08,
        52.34,
        13.76,
        52.68,
    )
    target_crs: str = "EPSG:25833"
    target_resolution: int = 10
    default_date: str = "2024-06-29"

    model_config = {"env_prefix": "BERLIN_LST_"}


settings = Settings()
