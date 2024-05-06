class Logger:
    def __init__(self, dirrectory: str):
        self.dir = dirrectory
        open(dirrectory, 'w').close()
        print("Logging start")

    def log(self, message: str):
        print("Logger: " + message)
        with open(self.dir, "a") as file:
            file.write(message + "\n")