from fastapi import FastAPI, Depends
import uvicorn
from enum import IntEnum
from typing import List, Optional
from pydantic import BaseModel, Field
app = FastAPI()

def get_todo_list():
    return  [
    Todo(todo_id=1, todo_name="coding", todo_description="code every data for 3 hours", priority=Priority.LOW),
    Todo(todo_id=2, todo_name= "gym", todo_description="code every data for 3 hours", priority=Priority.MEDIUM),
    Todo(todo_id=3, todo_name="play_with_dhiyaan", todo_description="code every data for 3 hours", priority=Priority.LOW),
    Todo(todo_id= 4, todo_name= "travel", todo_description="code every data for 3 hours", priority=Priority.HIGH)]

class Priority(IntEnum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description='Name of the todo')
    todo_description: str = Field(..., description="Description")
    priority: Priority = Field(default=Priority.LOW, description="priority")

class TodoCreate(BaseModel):
    pass

class Todo(TodoBase):
    todo_id: int = Field(..., description="unique id")

class TodoUpdate(BaseModel):
    todo_name: Optional[str] = Field(None, min_length=3, max_length=512, description="name of todo")
    todo_description: str = Field(None, description="Description")
    priority: Priority = Field(None, description="priority")



@app.get("/")
def hello_world():
    return "Hello world this is my fast api project"

@app.get("/todo_list")
def get_todo_list(to_do_list: List[Todo] = Depends(get_todo_list)):
    return to_do_list

@app.get("/todo_list")
def get_todo_list(to_do_list: List[Todo]=Depends(get_todo_list)):
    return to_do_list

@app.get("/todo_list/{todo_id}", response_model=Todo)
def get_todo_with_id(todo_id: int, to_do_list: List[Todo] = Depends(get_todo_list)):
    for todo in  to_do_list:
        if todo.todo_id == todo_id:
            return todo

@app.post("/todo_list/add_todos", response_model=Todo)
def create_todo(todo: dict, to_do_list:List[Todo] = Depends(get_todo_list)):
    new_todo_id = max(to_do['todo_id'] for to_do in to_do_list) + 1
    to_do_list.append(Todo(todo_id = new_todo_id,todo_name=  todo['todo_name'], todo_description = todo['todo_description']))
    return to_do_list

if __name__ == "__main__":
    uvicorn.run(app, port=8081, host="0.0.0.0")

