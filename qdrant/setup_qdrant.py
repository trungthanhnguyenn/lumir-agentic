import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path and load .env
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Ensure environment variables are loaded from project root .env
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

try:
    from module.rag_orchestrator import RAGOrchestratorFactory
    from module.database.qdrant_manager import QdrantManager
    print("RAG modules imported successfully")
except ImportError as e:
    print(f"Error importing RAG modules: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


def check_qdrant_installation():
    """Check if Qdrant is installed"""
    try:
        # Check qdrant-client
        import qdrant_client
        print("qdrant-client already installed")
        return True
    except ImportError:
        print("qdrant-client not found")
        return False


def install_qdrant():
    """Install Qdrant client"""
    try:
        print("Installing qdrant-client...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "qdrant-client"
        ])
        print("qdrant-client installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install qdrant-client: {e}")
        return False


def start_qdrant_server(port: int = 1237, data_dir: str = "./qdrant_data"):
    """Start Qdrant server using Docker"""
    try:
        print(f"Starting Qdrant server using Docker on port {port}...")
        
        # Check if Qdrant is running
        if check_qdrant_health(port):
            print(f"Qdrant server already running on port {port}")
            return True
        
        # Check Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Docker not available. Please install Docker first.")
            return False
        
        # Start Qdrant container
        project_root = Path(__file__).parent.parent
        docker_compose_file = project_root / "qdrant" / "docker-compose.yml"
        
        if not docker_compose_file.exists():
            print("Docker Compose file not found")
            return False
        
        print("Starting Qdrant container...")
        process = subprocess.Popen(
            ["docker", "compose", "-f", str(docker_compose_file), "up", "-d"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for container to start
        print("⏳ Waiting for Qdrant container to start...")
        time.sleep(10)
        
        if check_qdrant_health(port):
            print(f"Qdrant server started successfully via Docker on port {port}")
            return True
        else:
            print("Failed to start Qdrant server via Docker")
            return False
            
    except Exception as e:
        print(f"Error starting Qdrant server: {e}")
        return False


def stop_qdrant_server():
    """Stop Qdrant server Docker container"""
    try:
        print("Stopping Qdrant server...")
        
        project_root = Path(__file__).parent.parent
        docker_compose_file = project_root / "qdrant" / "docker-compose.yml"
        
        if not docker_compose_file.exists():
            print("Docker Compose file not found")
            return False
        
        # Stop container
        process = subprocess.run(
            ["docker", "compose", "-f", str(docker_compose_file), "down"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            print("Qdrant server stopped successfully")
            return True
        else:
            print(f"Failed to stop Qdrant server: {process.stderr}")
            return False
            
    except Exception as e:
        print(f"Error stopping Qdrant server: {e}")
        return False


def check_docker_container_status():
    """Check the status of Qdrant Docker container"""
    try:
        project_root = Path(__file__).parent.parent
        docker_compose_file = project_root / "qdrant" / "docker-compose.yml"
        
        if not docker_compose_file.exists():
            print("Docker Compose file not found")
            return False
        
        # Check container status
        process = subprocess.run(
            ["docker", "compose", "-f", str(docker_compose_file), "ps"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            print("Qdrant container status:")
            print(process.stdout)
            return True
        else:
            print(f"Failed to check container status: {process.stderr}")
            return False
            
    except Exception as e:
        print(f"Error checking container status: {e}")
        return False


def check_qdrant_health(port: int = 1237) -> bool:
    """Check if Qdrant server is running"""
    try:
        # Use /collections endpoint instead of /health
        response = requests.get(f"http://localhost:{port}/collections", timeout=5)
        if response.status_code == 200:
            return True
    except requests.RequestException:
        pass
    return False


def setup_rag_system():
    """Setup RAG system"""
    try:
        print("Setting up RAG system...")
        
        # Create RAG orchestrator
        orchestrator = RAGOrchestratorFactory.create_optimal_orchestrator()
        try:
            embed_info = orchestrator.embedding_manager.get_model_info()
            print(f"Embedding selected: {embed_info.get('provider')} - {embed_info.get('model_name')}")
        except Exception:
            pass
        
        # Setup collections
        if orchestrator.setup_rag_system():
            print("RAG system setup completed")
            return orchestrator
        else:
            print("Failed to setup RAG system")
            return None
            
    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        return None


def process_documents(orchestrator):
    """Process documents"""
    try:
        print("📄 Processing documents...")
        
        # Check documents directory
        docs_dir = Path("trading_data/general_infor")
        if not docs_dir.exists():
            print(f"Documents directory not found: {docs_dir}")
            print("Please create the directory and add your trading documents")
            return False
        
        # Process documents
        result = orchestrator.process_documents()
        
        if result.get("success"):
            print("Documents processed successfully")
            print(f"Total chunks: {result.get('total_chunks', 0)}")
            print(f"Total embeddings: {result.get('total_embeddings', 0)}")
            return True
        else:
            print(f"Document processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Error processing documents: {e}")
        return False


def test_rag_system(orchestrator):
    """Test RAG system"""
    try:
        print("Testing RAG system...")
        
        # Test simple query
        test_queries = [
            "Cách giao dịch hiệu quả?",
            "Tính năng của LUMIR-AI?",
            "Lộ trình huấn luyện trading?",
            "Lợi ích của platform?"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            
            from module.rag_orchestrator import RAGQuery
            
            rag_query = RAGQuery(
                text=query,
                language="vi",
                limit=3
            )
            
            response = orchestrator.query_rag(rag_query)
            
            if response.total_results > 0:
                print(f"Found {response.total_results} results")
                print(f"Collection used: {response.collection_used}")
                print(f"⚡ Processing time: {response.processing_time:.2f}s")
                
                # Display first result
                if response.results:
                    first_result = response.results[0]
                    print(f"Top result (score: {first_result.score:.3f}):")
                    content = first_result.payload.get("content", "")[:200]
                    print(f"   {content}...")
            else:
                print("No results found")
        
        print("\nRAG system test completed")
        return True
        
    except Exception as e:
        print(f"Error testing RAG system: {e}")
        return False


def main():
    """Main function"""
    import sys
    
    # Process command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--status":
            print("Checking Qdrant container status...")
            check_docker_container_status()
            return
        elif arg == "--stop":
            print("Stopping Qdrant server...")
            stop_qdrant_server()
            return
        elif arg == "--start":
            print("Starting Qdrant server...")
            start_qdrant_server()
            return
        elif arg == "--help":
            print("LUMIR RAG System Setup - Usage:")
            print("  python qdrant/setup_qdrant.py          # Full setup")
            print("  python qdrant/setup_qdrant.py --start  # Start Qdrant only")
            print("  python qdrant/setup_qdrant.py --stop   # Stop Qdrant only")
            print("  python qdrant/setup_qdrant.py --status # Check container status")
            print("  python qdrant/setup_qdrant.py --help   # Show this help")
            return
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            return
    
    print("LUMIR RAG System Setup")
    print("=" * 50)
    
    # Step 1: Check and install dependencies
    print("\nStep 1: Checking dependencies...")
    if not check_qdrant_installation():
        if not install_qdrant():
            print("Cannot proceed without qdrant-client")
            return
    
    # Step 2: Start Qdrant server
    print("\nStep 2: Starting Qdrant server...")
    if not start_qdrant_server():
        print("Cannot proceed without Qdrant server")
        return
    
    # Step 3: Setup RAG system
    print("\nStep 3: Setting up RAG system...")
    orchestrator = setup_rag_system()
    if not orchestrator:
        print("Cannot proceed without RAG system")
        return
    
    # Step 4: Process documents
    print("\nStep 4: Processing documents...")
    if not process_documents(orchestrator):
        print("Document processing failed, but continuing with test...")
    
    # Step 5: Test RAG system
    print("\nStep 5: Testing RAG system...")
    test_rag_system(orchestrator)
    
    # Bước 6: Hiển thị trạng thái
    print("\nStep 6: System status...")
    try:
        status = orchestrator.get_system_status()
        print(f"System status: {status.get('status', 'unknown')}")
        
        if 'collections' in status:
            print("Collections:")
            for name, info in status['collections'].items():
                if isinstance(info, dict) and 'points_count' in info:
                    print(f"   {name}: {info['points_count']} points")
                else:
                    print(f"   {name}: {info}")
        
        if 'documents' in status:
            docs = status['documents']
            print(f"Documents: {docs.get('total_files', 0)} files")
            if 'document_types' in docs:
                for doc_type, count in docs['document_types'].items():
                    print(f"   {doc_type}: {count}")
    
    except Exception as e:
        print(f"Error getting system status: {e}")
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Add your trading documents to 'trading_data/general_infor/'")
    print("2. Run this script again to process documents")
    print("3. Use the RAG system in your application")
    print("\nQdrant UI: http://localhost:1237/dashboard")
    print("Qdrant API: http://localhost:1237")
    print("\nDocker management commands:")
    print("   - Check container status: python qdrant/setup_qdrant.py --status")
    print("   - Stop Qdrant: python qdrant/setup_qdrant.py --stop")
    print("   - Start Qdrant: python qdrant/setup_qdrant.py --start")


if __name__ == "__main__":
    main()
