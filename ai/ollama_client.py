"""Ollama client with serving and model management capabilities"""

import json
import time
import subprocess
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import threading
from dataclasses import dataclass

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OllamaModel:
    """Ollama model information"""
    name: str
    size: str
    modified: str
    digest: str
    details: Dict[str, Any]


@dataclass
class OllamaResponse:
    """Ollama API response"""
    content: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """Enhanced Ollama client with model management and serving capabilities"""
    
    def __init__(self, 
                 host: str = "localhost:11434",
                 model_name: str = "llama2",
                 auto_start: bool = True,
                 auto_pull: bool = True):
        """Initialize Ollama client"""
        self.host = host
        self.base_url = f"http://{host}"
        self.model_name = model_name
        self.auto_start = auto_start
        self.auto_pull = auto_pull
        
        # Server management
        self.server_process = None
        self.is_server_running = False
        
        # Model cache
        self.available_models = {}
        self.model_info_cache = {}
        
        logger.info(f"Ollama client initialized for {host} with model {model_name}")
        
        # Auto-initialize if requested
        if auto_start:
            self.ensure_server_running()
        
        if auto_pull:
            self.ensure_model_available(model_name)
    
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed on the system"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Ollama installed: {result.stdout.strip()}")
                return True
            else:
                logger.warning("Ollama command failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Ollama not found or not responding: {e}")
            return False
    
    def is_server_accessible(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.debug(f"Server not accessible: {e}")
            return False
    
    def start_server(self, background: bool = True) -> bool:
        """Start Ollama server"""
        try:
            if not self.check_ollama_installed():
                logger.error("Ollama is not installed. Please install Ollama first.")
                return False
            
            if self.is_server_accessible():
                logger.info("Ollama server is already running")
                self.is_server_running = True
                return True
            
            logger.info("Starting Ollama server...")
            
            if background:
                # Start server in background
                self.server_process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait a moment for server to start
                time.sleep(3)
                
                # Check if server is now accessible
                if self.is_server_accessible():
                    self.is_server_running = True
                    logger.info("Ollama server started successfully")
                    return True
                else:
                    logger.error("Failed to start Ollama server")
                    return False
            else:
                # Start server in foreground (blocking)
                result = subprocess.run(['ollama', 'serve'], 
                                      capture_output=True, text=True)
                return result.returncode == 0
                
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop Ollama server"""
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                self.server_process = None
                logger.info("Ollama server stopped")
            
            self.is_server_running = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Ollama server: {e}")
            return False
    
    def ensure_server_running(self) -> bool:
        """Ensure Ollama server is running"""
        if self.is_server_accessible():
            self.is_server_running = True
            return True
        
        return self.start_server()
    
    def list_available_models(self) -> List[OllamaModel]:
        """List all available models"""
        try:
            if not self.ensure_server_running():
                return []
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get('models', []):
                model = OllamaModel(
                    name=model_data.get('name', ''),
                    size=model_data.get('size', ''),
                    modified=model_data.get('modified_at', ''),
                    digest=model_data.get('digest', ''),
                    details=model_data.get('details', {})
                )
                models.append(model)
                self.available_models[model.name] = model
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        models = self.list_available_models()
        return any(model.name.startswith(model_name) for model in models)
    
    def pull_model(self, model_name: str, show_progress: bool = True) -> bool:
        """Pull/download a model"""
        try:
            if not self.ensure_server_running():
                return False
            
            logger.info(f"Pulling model: {model_name}")
            
            # Use streaming API for progress
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            total_size = 0
            downloaded = 0
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        if 'total' in data and 'completed' in data:
                            total_size = data['total']
                            downloaded = data['completed']
                            
                            if show_progress and total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Downloading {model_name}: {progress:.1f}%")
                        
                        if data.get('status') == 'success':
                            logger.info(f"Successfully pulled model: {model_name}")
                            return True
                            
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pull if necessary"""
        if self.is_model_available(model_name):
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Model {model_name} not found, attempting to pull...")
        return self.pull_model(model_name)
    
    def generate_response(self, 
                         prompt: str, 
                         model: Optional[str] = None,
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> Optional[OllamaResponse]:
        """Generate response from Ollama model"""
        
        if model is None:
            model = self.model_name
        
        try:
            if not self.ensure_server_running():
                logger.error("Ollama server is not running")
                return None
            
            if not self.ensure_model_available(model):
                logger.error(f"Model {model} is not available")
                return None
            
            # Prepare request data
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
            
            if max_tokens:
                request_data["options"]["num_predict"] = max_tokens
            
            # Make request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=120  # 2 minutes timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            return OllamaResponse(
                content=data.get('response', ''),
                model=data.get('model', model),
                created_at=data.get('created_at', ''),
                done=data.get('done', True),
                total_duration=data.get('total_duration'),
                load_duration=data.get('load_duration'),
                prompt_eval_count=data.get('prompt_eval_count'),
                eval_count=data.get('eval_count')
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None
    
    def generate_bookmark_summary(self, 
                                 title: str, 
                                 content: str, 
                                 url: str,
                                 max_length: int = 200) -> str:
        """Generate a concise summary for a bookmark"""
        
        system_prompt = f"""You are an expert at creating concise, informative summaries of web content. 
Create a summary that is exactly {max_length} characters or less. Focus on the key purpose, 
main features, and value proposition of the content."""
        
        prompt = f"""Summarize this bookmark in {max_length} characters or less:

Title: {title}
URL: {url}
Content: {content[:1000]}  # Limit content to avoid token limits

Summary:"""
        
        try:
            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=100
            )
            
            if response and response.content:
                summary = response.content.strip()
                # Ensure it's within length limit
                if len(summary) > max_length:
                    summary = summary[:max_length-3] + "..."
                return summary
            else:
                return f"Summary for {title}"
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Summary for {title}"
    
    def suggest_categories(self, 
                          title: str, 
                          content: str, 
                          url: str,
                          existing_categories: List[str]) -> List[str]:
        """Suggest categories for a bookmark"""
        
        categories_text = ", ".join(existing_categories) if existing_categories else "None available"
        
        system_prompt = """You are an expert at categorizing web content. Suggest 1-3 most appropriate 
categories from the existing list, or suggest new categories if none fit well. Be concise and specific."""
        
        prompt = f"""Categorize this bookmark:

Title: {title}
URL: {url}
Content: {content[:800]}

Existing categories: {categories_text}

Suggest 1-3 most appropriate categories (comma-separated):"""
        
        try:
            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=50
            )
            
            if response and response.content:
                categories = [cat.strip() for cat in response.content.split(',')]
                return [cat for cat in categories if cat and len(cat) > 2][:3]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to suggest categories: {e}")
            return []
    
    def generate_smart_tags(self, 
                           title: str, 
                           content: str, 
                           url: str,
                           existing_tags: List[str] = None) -> List[str]:
        """Generate intelligent tags for a bookmark"""
        
        if existing_tags is None:
            existing_tags = []
        
        existing_text = ", ".join(existing_tags) if existing_tags else "None"
        
        system_prompt = """You are an expert at generating relevant, specific tags for web content. 
Generate 3-7 concise, descriptive tags that capture the key topics, technologies, and purposes. 
Avoid generic tags and focus on specific, searchable terms."""
        
        prompt = f"""Generate smart tags for this bookmark:

Title: {title}
URL: {url}
Content: {content[:800]}

Existing tags: {existing_text}

Generate 3-7 specific, relevant tags (comma-separated):"""
        
        try:
            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=80
            )
            
            if response and response.content:
                tags = [tag.strip() for tag in response.content.split(',')]
                # Filter and clean tags
                clean_tags = []
                for tag in tags:
                    if tag and len(tag) > 2 and len(tag) < 30:
                        clean_tags.append(tag.title())
                
                return clean_tags[:7]  # Limit to 7 tags
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to generate smart tags: {e}")
            return []
    
    def extract_key_concepts(self, content: str, limit: int = 10) -> List[str]:
        """Extract key concepts from content"""
        
        system_prompt = f"""You are an expert at extracting key concepts from text. 
Extract the {limit} most important concepts, topics, or themes. Focus on specific, 
meaningful terms rather than generic words."""
        
        prompt = f"""Extract key concepts from this content:

{content[:1200]}

Extract {limit} key concepts (comma-separated):"""
        
        try:
            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            if response and response.content:
                concepts = [concept.strip() for concept in response.content.split(',')]
                return [concept for concept in concepts if concept and len(concept) > 2][:limit]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract key concepts: {e}")
            return []
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get Ollama server information"""
        try:
            if not self.is_server_accessible():
                return {"status": "not_accessible", "models": []}
            
            models = self.list_available_models()
            
            return {
                "status": "running",
                "host": self.host,
                "base_url": self.base_url,
                "models_count": len(models),
                "models": [model.name for model in models],
                "default_model": self.model_name,
                "server_process_running": self.server_process is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.server_process:
            try:
                self.stop_server()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to cleanup resources"""
        self.cleanup()