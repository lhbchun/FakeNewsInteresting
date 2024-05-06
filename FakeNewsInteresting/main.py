from traceback import format_exc
import datetime

from verboseLogger import Logger
logger = Logger("log.txt")
logger.log("Logging start on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

logger.log("Importing Hydrate")
import hydrate
logger.log("Importing preprocess")
import preprocess
logger.log("Importing process")
import process


try:
    preprocess.express(logger)
    process.express(logger)
except Exception as e:
    logger.log(format_exc())
    raise e
    
