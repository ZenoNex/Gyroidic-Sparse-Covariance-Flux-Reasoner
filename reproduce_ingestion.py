
import json
import ijson
import os

def test_ijson_parsing():
    print("Testing ijson parsing...")
    
    # optimize for top-level list
    data_list = [{"id": 1, "text": "sample 1"}, {"id": 2, "text": "sample 2"}]
    filename_list = "test_list.json"
    with open(filename_list, 'w') as f:
        json.dump(data_list, f)
        
    # optimize for wrapping object
    data_obj = {"conversations": [{"id": 1, "text": "sample 1"}, {"id": 2, "text": "sample 2"}]}
    filename_obj = "test_obj.json"
    with open(filename_obj, 'w') as f:
        json.dump(data_obj, f)
        
    print(f"\n1. Testing Top-Level List: {filename_list}")
    try:
        with open(filename_list, 'r') as f:
            # Current code uses 'item'
            objects = ijson.items(f, 'item')
            count = 0
            for o in objects:
                count += 1
                print(f"   Found: {o}")
            print(f"   Total items with prefix 'item': {count}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    try:
        with open(filename_list, 'r') as f:
            # Try empty prefix? No, 'item' should work for list elements.
            # Try '' (root)
            objects = ijson.items(f, '')
            for o in objects:
                print(f"   Root object type: {type(o)}")
    except Exception as e:
        print(f"   ❌ Root Error: {e}")

    print(f"\n2. Testing Wrapped Object: {filename_obj}")
    try:
        with open(filename_obj, 'r') as f:
            # Current code uses 'item' -> This will fail for wrapped object
            objects = ijson.items(f, 'item')
            count = 0
            for o in objects:
                count += 1
            print(f"   Total items with prefix 'item': {count}")
    except Exception as e:
         print(f"   ❌ Error: {e}")

    try:
        with open(filename_obj, 'r') as f:
             # Try 'conversations.item'
            objects = ijson.items(f, 'conversations.item')
            count = 0
            for o in objects:
                count += 1
            print(f"   Total items with prefix 'conversations.item': {count}")
    except Exception as e:
         print(f"   ❌ Error: {e}")
         
    # Clean up
    if os.path.exists(filename_list): os.remove(filename_list)
    if os.path.exists(filename_obj): os.remove(filename_obj)

if __name__ == "__main__":
    test_ijson_parsing()
