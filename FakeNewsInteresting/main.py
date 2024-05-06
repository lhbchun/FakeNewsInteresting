import datetime

from verboseLogger import Logger
logger = Logger("log.txt")
logger.log("Logging start on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

import hydrate
import preprocess
import process


try:
    preprocess.express(logger)
    process.express(logger)
except Exception as e:
    logger.log("Error: " + str(e))
    raise e
    
