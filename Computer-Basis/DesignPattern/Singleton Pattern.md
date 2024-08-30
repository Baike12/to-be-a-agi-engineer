### 单例模式是什么
- 一个类只有一个实例
- 全局访问点：严格控制对这个实例的访问
- 优点
	- 节省资源
	- 懒加载：只有需要的时候才创建实例
- 两种模式
	- 懒汉式：需要时才创建
	- 饿汉式：类加载时就完成实例创建，不管后面用没用到
### 适用场景
- 全局只有一个需要共享的实例
	- 连接池管理
	- 配置管理
### 实现
- 构造函数私有：防止外部构造
- 私有的静态实例变量：保存唯一的实例
- 公有的方法：给外部访问
#### c++实现
```c++

```
#### go实现
```go
package singleton

import (
	"fmt"
	"sync"
)

type ShoppingCartManager struct {
	cart map[string]int
	keys []string
	mu   sync.Mutex
}

var onece sync.Once
var scmInstance *ShoppingCartManager // 私有的实例

func GetInstance() *ShoppingCartManager {
	onece.Do(func() { // 只实例化一次
		scmInstance = &ShoppingCartManager{
			cart: make(map[string]int),
		}
	})
	return scmInstance
}

func (scm *ShoppingCartManager) AddToCart(itemName string, n int) {
	scm.mu.Lock()
	defer scm.mu.Unlock()

	if _, exists := scmInstance.cart[itemName]; !exists {
		scm.keys = append(scm.keys, itemName)
	}
	scm.cart[itemName] += n
}

func (scm *ShoppingCartManager) ViewCart() {
	scm.mu.Lock()
	defer scm.mu.Unlock()

	for _, item := range scm.keys {
		n := scm.cart[item]
		fmt.Println("goods:", item, "---", "quantity:", n)
	}
}

func Singleton() {
	cart := singleton.GetInstance()
	scanner := bufio.NewScanner(os.Stdin)

	for scanner.Scan() {
		input := scanner.Text()
		if input == "" {
			break
		}
		parts := strings.Fields(input)
		item := parts[0]
		nums := 0
		if len(parts) > 1 {
			fmt.Sscanf(parts[1], "%d", &nums)
		}
		cart.AddToCart(item, nums)
	}
	cart.ViewCart()
}
```