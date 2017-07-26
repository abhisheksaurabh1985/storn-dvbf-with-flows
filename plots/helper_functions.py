def sliceFrom3DTensor(tensor, idx):
    """
    Returns a 2D slice from a 3D tensor slicing along the third dimension.
    Assumes the tensor to be of shape (T*B*D).
    
    Args:
        tensor: 3D tensor to be sliced.
        idx: Index of the 
        
    Returns:
        tensor_2D:
    """
    tensor_2D = tensor[:, :, idx]
    return tensor_2D

