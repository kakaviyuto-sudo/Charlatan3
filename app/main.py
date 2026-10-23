from config.env import load_settings
from app.runtime import Runtime

def main() -> None:
    settings = load_settings()
    runtime = Runtime(settings)
    try:
        runtime.start()
        runtime.run_forever()
    except KeyboardInterrupt:
        print("Shutdown requested by user.")
    finally:
        runtime.stop()

if __name__ == "__main__":
    main()