# Bad example (violates SRP)
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def save_to_database(self):
        print(f"Saving user {self.name} to database.")

    def send_email(self, message):
        print(f"Sending email to {self.email}: {message}")

# Good example (adheres to SRP)
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user):
        print(f"Saving user {user.name} to database.")

class EmailService:
    def send_email(self, user, message):
        print(f"Sending email to {user.email}: {message}")

# liskov substritution

#object of super class should be replaceable with object of subclass without affecting the functionalityclass Bird:
    def fly(self):
        print("Bird is flying")

class Ostrich(Bird):
    def fly(self):
        raise NotImplementedError("Ostriches cannot fly")

def make_bird_fly(bird: Bird):
    bird.fly()

# Usage
bird = Bird()
make_bird_fly(bird)  # Output: Bird is flying

ostrich = Ostrich()
# make_bird_fly(ostrich) # This would raise NotImplementedError, violating LSP

class FlyingBird:
    def fly(self):
        print("Flying bird is flying")

class Sparrow(FlyingBird):
    def fly(self):
        print("Sparrow is flying gracefully")

class NonFlyingBird:
    def walk(self):
        print("Non-flying bird is walking")

class Ostrich(NonFlyingBird):
    def walk(self):
        print("Ostrich is walking majestically")

def make_flying_bird_fly(bird: FlyingBird):
    bird.fly()

def make_non_flying_bird_walk(bird: NonFlyingBird):
    bird.walk()

# Usage
sparrow = Sparrow()
make_flying_bird_fly(sparrow)  # Output: Sparrow is flying gracefully

ostrich = Ostrich()
make_non_flying_bird_walk(ostrich) # Output: Ostrich is walking majestically
# correct program

# dependency inversion principle

class Database(ABC):
    @abstractmethod
    def save(self, data):
        pass

class MySQLDatabase(Database):
    def save(self, data):
        print("Saving to MySQL")

class DataManager:
    def __init__(self, db: Database):
        self.db = db
    def save_data(self, data):
        self.db.save(data)

db = MySQLDatabase()
manager = DataManager(db)
manager.save_data("info")

