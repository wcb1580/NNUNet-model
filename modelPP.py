import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
"""UNET++ with nested skip connection, deep supervision and 4 layers

"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, middle_channels,out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, 1, 1,dilation = 1, padding_mode='zeros',bias=False)
        self.ba1 = nn.BatchNorm2d(middle_channels)
        self.drop = nn.Dropout(p=0.2)
        # self.pool = nn.MaxPool2d()
        self.re = nn.ReLU(inplace=True)
            #nn.Dropout(p=0.4),
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, 1, 1, dilation = 1,padding_mode='zeros',bias=False)
        self.ba2 = nn.BatchNorm2d(out_channels)
        nn.Dropout(p=0.2),
        self.pool = nn.AvgPool2d(3)

        

    def forward(self, x):
        out = self.conv1(x)
        out = self.ba1(out)
        out = self.drop(out)
        out = self.re(out)

        out = self.conv2(out)
        out = self.ba2(out)
        out = self.re(out)
        # out = self.pool9)
        # out = self.pool(out)
        return out
class UNET(nn.Module):
    def __init__(
            self, in_channels=3,out_channels=1, features=[64, 128, 256, 512,1024], deep_supervision = False                                                                                                                 
    ):
        super(UNET, self).__init__()
        self.deep_supervision = deep_supervision
        self.ups = nn.Upsample(scale_factor=2,mode = 'bicubic',align_corners=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 =  DoubleConv(in_channels,features[0],features[0])
        self.conv1_0 = DoubleConv(features[0],features[1],features[1])
        self.conv2_0 = DoubleConv(features[1],features[2],features[2])
        self.conv3_0 = DoubleConv(features[2],features[3],features[3])
        self.conv4_0 = DoubleConv(features[3],features[4],features[4])

        # Intermediate nodes used to preserve feature maps
        self.conv0_1 = DoubleConv(features[1]*1+features[0],features[0],features[0])
        self.conv1_1 = DoubleConv(features[2]*1+features[1],features[1],features[1])
        self.conv2_1 = DoubleConv(features[3]*1+features[2],features[2],features[2])
        self.conv3_1 = DoubleConv(features[4]*1+features[3],features[3],features[3])

        self.conv0_2 = DoubleConv(features[0]*2+features[1],features[0],features[0])
        self.conv1_2 = DoubleConv(features[1]*2+features[2],features[1],features[1])
        self.conv2_2 = DoubleConv(features[2]*2+features[3],features[2],features[2])

        self.conv0_3 = DoubleConv(features[0]*3+features[1],features[0],features[0])
        self.conv1_3 = DoubleConv(features[1]*3+features[2],features[1],features[1])

        self.conv0_4 = DoubleConv(features[0]*4+features[1],features[0],features[0])

        
        if deep_supervision:
            self.final1 = nn.Conv2d(features[0],out_channels,kernel_size=1)
            self.final2 = nn.Conv2d(features[0],out_channels,kernel_size=1)
            self.final3 = nn.Conv2d(features[0],out_channels,kernel_size=1)
            self.final4 = nn.Conv2d(features[0],out_channels,kernel_size=1)
        else:
            self.final = nn.Conv2d(features[0],out_channels,kernel_size=1)

        """# Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)"""

    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.concat([x0_0,self.ups(x1_0)],1))
        

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.concat([x1_0,self.ups(x2_0)],1))
        x0_2 = self.conv0_2(torch.concat([x0_0,x0_1,self.ups(x1_1)],1))
        
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.concat([x2_0,self.ups(x3_0)],1))
        x1_2 = self.conv1_2(torch.concat([x1_0,x1_1,self.ups(x2_1)],1))

        # print('test')
        x0_3 = self.conv0_3(torch.concat([x0_0,x0_1,x0_2,self.ups(x1_2)],1))
        #print('tested')
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.concat([x3_0,self.ups(x4_0)],1))
        x2_2 = self.conv2_2(torch.concat([x2_1,x2_0,self.ups(x3_1)],1))
        x1_3 = self.conv1_3(torch.concat([x1_0,x1_1,x1_2,self.ups(x2_2)],1))
        x0_4 = self.conv0_4(torch.concat([x0_0,x0_1,x0_2,x0_3,self.ups(x1_3)],1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
       
            # Now, you can safely combine the tensors
            Combined = output1 + output2 + output3 + output4
            Combined /= 4
            
            return Combined, [output1,output2,output3,output4]
        else:
            output = self.final(x0_4)
            return output
        
        
        

        """skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)"""

def test():
    # Create a random input tensor of size [batch_size, channels, height, width].
    x = torch.randn((1, 3, 256, 256))  # Example input

    # Initialize the UNET model
    model = UNET(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024], deep_supervision=True)

    # Forward pass of the model with deep supervision enabled
    combined_output, deep_sup_outputs = model(x)
    
    print("Combined Output Shape:", combined_output.shape)
    print("Deep Supervision Outputs Shapes:")
    for i, output in enumerate(deep_sup_outputs, 1):
        print(f"Output {i} Shape:", output.shape)

    # Check for expected output shapes
    assert combined_output.shape == (1, 1, 256, 256), "Mismatch in combined output shape"
    for output in deep_sup_outputs:
        assert output.shape == (1, 1, 256, 256), "Mismatch in deep supervision output shape"

    # Test model with deep supervision disabled
    model_deep_sup_disabled = UNET(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024], deep_supervision=False)
    output_without_deep_sup = model_deep_sup_disabled(x)
    print("Output Shape without Deep Supervision:", output_without_deep_sup.shape)
    assert output_without_deep_sup.shape == (1, 1, 256, 256), "Mismatch in output shape without deep supervision"

if __name__ == "__main__":
    test()
    
    
