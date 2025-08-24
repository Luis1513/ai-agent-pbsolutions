"""
Application Settings Configuration

This module manages all application configuration settings using Pydantic Settings.
It handles environment variables, configuration validation, and provides a centralized
way to access all application settings throughout the system.

Dependencies:
    - Pydantic Settings for configuration management
    - Python-dotenv for environment variable loading
    - Pathlib for file path handling
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

#*****************************************************************************************************#
# Repository root path detection and environment file configuration
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_FILE_PATH = REPOSITORY_ROOT / ".env"

# Load environment variables from .env file explicitly (useful for development)
load_dotenv(ENVIRONMENT_FILE_PATH)
#*****************************************************************************************************#



#*****************************************************************************************************#
class Settings(BaseSettings):
    """
    Application settings configuration class.
    
    This class manages all application configuration using Pydantic Settings.
    It automatically loads environment variables and provides validation.
    
    Attributes:
        openai_api_key: OpenAI API key for authentication
        openai_model: OpenAI model for text generation
        openai_embedding_model: OpenAI model for text embeddings
        pinecone_api_key: Pinecone API key for vector database
        pinecone_index: Pinecone index name for vector storage
        pinecone_cloud: Cloud provider for Pinecone (default: aws)
        pinecone_region: Region for Pinecone service (default: us-east-1)
        env: Application environment (default: dev)
    """
    
    # Pydantic-settings configuration for environment file handling
    model_config = SettingsConfigDict(
        env_file=str(ENVIRONMENT_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra environment variables
    )

    # OpenAI Configuration
    openai_api_key: str = Field(
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for authentication and API access"
    )
    
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_MODEL",
        description="OpenAI model for text generation and chat completion"
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
        description="OpenAI model for generating text embeddings"
    )
    

    # Pinecone Vector Database Configuration
    pinecone_api_key: str = Field(
        validation_alias="PINECONE_API_KEY",
        description="Pinecone API key for vector database access"
    )
    
    pinecone_index: str = Field(
        default="pb-rag",
        validation_alias="PINECONE_INDEX",
        description="Name of the Pinecone index for vector storage"
    )
    
    pinecone_cloud: str = Field(
        default="aws",
        validation_alias="PINECONE_CLOUD",
        description="Cloud provider hosting Pinecone service"
    )
    
    pinecone_region: str = Field(
        default="us-east-1",
        validation_alias="PINECONE_REGION",
        description="Geographic region for Pinecone service"
    )
   

    # Application Configuration
    env: str = Field(
        default="dev",
        validation_alias="ENV",
        description="Application environment (dev, staging, production)"
    )
#*****************************************************************************************************#



#*****************************************************************************************************#
# Global settings instance
settings = Settings()
print(f"[OK] Settings loaded: env={settings.env}, model={settings.openai_model}")
#*****************************************************************************************************#
