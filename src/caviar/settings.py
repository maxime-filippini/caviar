from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Mostly used to easily read API keys from environment variables or .env file.
    """

    model_config = SettingsConfigDict(env_file=".env")
    MARKET_API_KEY: SecretStr


SETTINGS = Settings()  # pyright: ignore[reportCallIssue]
