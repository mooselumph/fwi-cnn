import matplotlib.pyplot as plt

import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor



def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_speeds(speeds,speeds_pred):
    """Create a pyplot plot and save to buffer."""
    
    
    f = plt.figure()
    plt.plot(speeds,label='True Speeds')
    plt.plot(speeds_pred,label='Predicted Speeds')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    
    #image = image.detach().cpu().numpy().squeeze().transpose(1,2,0)
    image = image[:,0:3,:,:]
    
    plt.close(f)
    
    return image


def plot_amplitudes(amps):
    """Create a pyplot plot and save to buffer."""
    
    f = plt.figure()
    
    amps = amps.detach().cpu().numpy()
    amps = amps[0,0,:,:].squeeze()
    
    plt.imshow(amps,cmap='gray')
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    
    image = image[:,0:3,:,:]
    
    plt.close(f)
    
    return image