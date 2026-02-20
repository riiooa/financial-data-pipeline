import psycopg2
import time
import logging

from sqlalchemy import exc
from src.utils.logging_utils import retry, timer, db_transaction, setup_logging

# Initialize logging so that retry messages appear in the terminal.
setup_logging()
logger = logging.getLogger(__name__)

# Using our custom retry decorator
# max_attempts=3 means: 1 run + 2 retries
@retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(ValueError,))
def flaky_function():
    """The simulation function was intentionally set to fail"""
    print("\n[Executing] flaky_function is running...")
    raise ValueError("Simulation error: Connection lost!")

def test_retry_mechanism():
    print("=== Starting the Error Handling Test (Retry) ===")
    start_time = time.time()

    try:
        with timer("Testing Flaky Function"):
            flaky_function()
    except ValueError as e:
        duration = time.time() - start_time
        print(f"\n[Final Result] Failed after all attempts: {e}")
        print(f"[Total Time]  Done in {duration:.2f} seconds")

    print("=== Test Done ===\n")

if __name__=="__main__":
    test_retry_mechanism()