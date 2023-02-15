import tkinter as tk

# initialize the window
root = tk.Tk()
root.title("South East Con Mapping 2023")

# initialize the add object section
add_obj_label = tk.Label(root, text="Add Object")
add_obj_label.grid(row=0, column=0, padx=5, pady=5)

label_entry = tk.Entry(root)
label_entry.grid(row=1, column=0, padx=5, pady=5)

x_entry = tk.Entry(root)
x_entry.grid(row=1, column=1, padx=5, pady=5)

y_entry = tk.Entry(root)
y_entry.grid(row=1, column=2, padx=5, pady=5)

# initialize the move robot section
move_robot_label = tk.Label(root, text="Move Robot")
move_robot_label.grid(row=3, column=0, padx=5, pady=5)

robot_x_entry = tk.Entry(root)
robot_x_entry.grid(row=4, column=0, padx=5, pady=5)

robot_y_entry = tk.Entry(root)
robot_y_entry.grid(row=4, column=1, padx=5, pady=5)

speed_entry = tk.Entry(root)
speed_entry.grid(row=4, column=2, padx=5, pady=5)

# initialize the list to store the objects
objects = []

# function to handle add object submit
def add_object():
    label = label_entry.get()
    x = x_entry.get()
    y = y_entry.get()

    objects.append({
        'label': label,
        'x': x,
        'y': y
    })

    print("Object Added: ", objects)

# function to handle move robot submit
def move_robot():
    x = robot_x_entry.get()
    y = robot_y_entry.get()
    speed = speed_entry.get()

    # update the x and y variables
    # ...

# initialize the submit buttons
add_obj_submit = tk.Button(root, text="Submit", command=add_object)
add_obj_submit.grid(row=2, column=1, padx=5, pady=5)

move_robot_submit = tk.Button(root, text="Submit", command=move_robot)
move_robot_submit.grid(row=5, column=1, padx=5, pady=5)

# run the main loop
root.mainloop()
