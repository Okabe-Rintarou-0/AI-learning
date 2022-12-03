# hw6

![](task.png)

## WGAN

https://arxiv.org/abs/1701.07875

参考链接：https://zhuanlan.zhihu.com/p/539156191

+ 原理
    
    `WGAN` 的 `Discriminator` 本质上就是要去拟合这样的一个 `Wasserstein Distance`。
    原先的 `GAN` 的 `Discriminator` 是在模拟 JS 散度，是一个分类问题。而 `WGAN` 则变成了一个回归问题，
    故而需要去掉最后的 `Sigmoid` 层。

+ 优化器选择 `RMSProp`

    之所以不用 `Adam` 是因为 loss 不稳定，冲量会加剧这一不稳定性。
    
    > RMSProp采用了指数加权移动平均(exponentially weighted moving average)。
    >
    > RMSProp比AdaGrad只多了一个超参数，其作用类似于动量(momentum)，其值通常置为0.9。
    >
    > RMSProp旨在加速优化过程，例如减少达到最优值所需的迭代次数，或提高优化算法的能力，例如获得更好的最终结果。

    ![](RMSProp.png)
    
+ 损失函数

    回归问题：
    
    ```python
    loss_G = -torch.mean(D(f_imgs))
    ```
    
+ Parameter Clipping

    `WGAN` 希望拟合出来的函数是满足 `1-Lipschitz` 条件的，也就是函数需要足够光滑。
    
    利用 `Clipping` 去近似这种光滑：限制参数的数值范围在 [-c, c] 之间，其中 `c` 是一个常数。
    

## WGAN-GP

https://arxiv.org/abs/1704.00028 

将较为粗糙的 `Clipping` 换为一个梯度的惩罚（类似于正则化）。会在生成的 `x_hat` 和真实的 `x` 之前采样一个 `x`，计算它的梯度，
并得到对应的惩罚值（梯度越接近 1，这种惩罚值越低）。因为 `1-Lipschitz` 条件也就是在说函数任意一点的梯度不能超过 1，但我们不可能把每一个点都考虑进去，所以是采样一个点，
对其做 `Gradient Penalty`。

![](WGAN-GP1.png)

损失函数：

![](WGAN-GP2.png)

## 效果

<table>
    <tr>
        <td>模型</td>
        <td>Epoch = 5</td>
        <td>Epoch = 25</td>
        <td>Epoch = 50</td>
        <td>结果</td>
    </tr>
    <tr>
        <td>DCGAN</td>
        <td><img src="DCGAN-epoch-5.jpg" alt=""/></td>
        <td><img src="DCGAN-epoch-25.jpg" alt=""/></td>
        <td><img src="DCGAN-epoch-50.png" alt=""/></td>
        <td><a href="DCGAN-result.jpg">DCGAN-result.jpg</a></td>
    </tr>
    <tr>
        <td>WGAN</td>
        <td><img src="" alt=""/></td>
        <td><img src="" alt=""/></td>
        <td><img src="" alt=""/></td>
        <td></td>
    </tr>
    <tr>
        <td>WGAN-GP</td>
        <td><img src="" alt=""/></td>
        <td><img src="" alt=""/></td>
        <td><img src="" alt=""/></td>
        <td></td>
    </tr>
</table>
