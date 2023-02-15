# Binary Search Tree operations in Python
 
 
# Create a node
class Node:
   def __init__(self, key,class_type,x_y):
       self.key = key
       self.left = None
       self.right = None
       self.class_type = class_type
       self.x_y = x_y
 
 
# Inorder traversal
def inorder(root):
   if root is not None:
       # Traverse left
       print("In the tree")
       inorder(root.left)
 
       # Traverse root
       print(str(root.key) + "->", end=' ')
 
       # Traverse right
       inorder(root.right)
 
def find_path(root):
   if root is not None:
       # Traverse left
       inorder(root.left)
 
       # Traverse root
       print(str(root.key) + "->", end=' ')
 
       # Traverse right
       inorder(root.right)
# Inorder traversal
def inorder_return(root):
   if root is not None:
        print(str(root.key) + "->", end=' ')
        # Traverse left
        inorder(root.left)
    
        # Traverse root

        print(str(root.key) + "->", end=' ')
    
        #Travesre right
        inorder(root.right)


 
def inorder_grouping(root,group,groups):
    print("Grouping")
    if root is not None:

        # Traverse left
        size = sum(1 for num in group)
        #If the group is full
        if size == 3:
            groups.append(group)
            group = []

            # Add pedastal to current group
        if root.class_type == "Pedastal":
            group.append(root.class_type)
        inorder(root.left)

        # Traverse root
        print(str(root.key) + "->", end=' ')
    
        # Traverse right
        inorder(root.right)

            
    groups.append(group)
    print("Group: ")
    print(groups)


 
# Insert a node
def insert(node, key, class_type,x_y):
 
   # Return a new node if the tree is empty
   if node is None:
       return Node(key,class_type,x_y)
 
   # Traverse to the right place and insert the node
   if key < node.key:
       node.left = insert(node.left, key,class_type,x_y)
   else:
       node.right = insert(node.right, key,class_type,x_y)
 
   return node
 
 
# Find the inorder successor
def minValueNode(node):
   current = node
 
   # Find the leftmost leaf0.0421
   while(current.left is not None):
       current = current.left
 
   return current
 
 
# Deleting a node
def deleteNode(root, key):
 
   # Return if the tree is empty
   if root is None:
       return root
 
   # Find the node to be deleted
   if key < root.key:
       root.left = deleteNode(root.left, key)
   elif(key > root.key):
       root.right = deleteNode(root.right, key)
   else:
       # If the node is with only one child or no child
       if root.left is None:
           temp = root.right
           root = None
           return temp
 
       elif root.right is None:
           temp = root.left
           root = None
           return temp
 
       # If the node has two children,
       # place the inorder successor in position of the node to be deleted
       temp = minValueNode(root.right)
 
       root.key = temp.key
 
       # Delete the inorder successor
       root.right = deleteNode(root.right, temp.key)
 
   return root
 
 
 
 

