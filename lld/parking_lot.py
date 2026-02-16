from enum import Enum
class VehicleType(Enum):
    MOTORCYCLE = 1
    CAR = 2
    LARGE = 2

class Ticket:
    def __init__(self, vehicle_id, sport_id, vehicle_type, entry_time):
        self.id = id
        self.vehicle_type = VehicleType
        self.sport_id = sport_id
        self.entry_time = entry_time

    def get_id(self):
        return self.id
    def get_vehicle_type(self):
        return self.vehicle_type
    def get_sport_id(self):
        return self.sport_id
    def get_entry_time(self):
        return self.entry_time

class ParkingSpot:
    def __init__(self, spot_id, spot_type):
        self.sport_id = spot_id
        self.spot_type = spot_type
        self.is_occupied = False

    def is_occupied(self):
        return self.is_occupied

    def get_sport_type(self):
        return self.spot_type

    def mark_occupied(self):
        self.is_occupied = True

    def unmark_occupied(self):
        self.is_occupied = False



class ParkingLot:
    def __init__(self, spots, rate):
        self.parking_spots = []
        self.active_tickets = {}
        self.rate = rate

    def enter(self):
        pass

    def exit(self, ticket_id, exit_time):
        pass
