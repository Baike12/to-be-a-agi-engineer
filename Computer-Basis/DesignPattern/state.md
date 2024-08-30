### state模式是什么
- 在对象内部状态改变时改变其行为，使得看起来就像改变了自身所属类一样
- 类似于状态机
### 适用场景
- 对象需要根据当前状态进行不同的行为，并且状态变更频繁
### 实现
#### 概念
- 主体：会保存对于具体状态的引用，以在收到操作后执行特定状态
- 状态接口
- 实际状态
#### 具体实现
- 自动售货机根据不同的状态执行不同的操作，但是自动售卖机本身的代码不受影响，只影响状态代码
	- 定义状态接口，接口中包含一系列方法
	- 定义具体状态，每个状态只能在特定的请求下工作，比如在没有商品的时候只能执行添加商品这一个操作，其他操作都会报错 
#### c++实现
```c++

```
#### go实现
```go
package state

import (
	"fmt"
	"log"
)

// 定义售货机
type VendingMachine struct {
	hasItem       State
	itemRequested State
	hasMoney      State
	noItem        State

	curState  State // 当前状态
	itemCount int
	itemPrice int
}

// 初始化一个售卖机
func newVendingMachine(itemCount, itemPrice int) *VendingMachine {
	v := &VendingMachine{
		itemCount: itemCount,
		itemPrice: itemPrice,
	}
	// 初始化一些状态
	v.noItem = &NoItemState{
		vendingMachine: v,
	}
	v.hasItem = &HasItemState{
		vendingMachine: v,
	}
	// 设置当前状态
	v.curState = v.noItem
	return v
}

func (v *VendingMachine) incrementItemCount(count int) {
	fmt.Printf("add %d items\n", count)
	v.itemCount += count
}

func (v *VendingMachine) setState(s State) {
	v.curState = s
}

// 售货机执行各种操作// 从当前状态
func (v *VendingMachine) addItem(count int) error {
	return v.curState.addItem(count)
}

func (v *VendingMachine) requestItem() error {
	return v.curState.requestItem()
}

// 定义状态
type State interface { // 状态定义一些操作
	addItem(int) error
	requestItem() error
	insertMoney(money int) error
	dispenseItem() error
}

// 定义没有商品的状态
type NoItemState struct {
	vendingMachine *VendingMachine // 每一个状态都引用一个售卖机
}

func (n *NoItemState) addItem(count int) error {
	n.vendingMachine.incrementItemCount(count)
	// 然后将状态设置为有商品状态：意思是状态更新在每一个状态中完成，售卖机无感知
	n.vendingMachine.setState(n.vendingMachine.hasItem)
	return nil
}

// 没有商品的情况下只能添加商品，其他动作都会报错
func (n *NoItemState) requestItem() error {
	return fmt.Errorf("there is no item")
}

func (n *NoItemState) insertMoney(money int) error {
	return fmt.Errorf("there is no item")
}

func (n *NoItemState) dispenseItem() error {
	return fmt.Errorf("there is no item")
}

// 有商品的状态：可以添加商品和请求商品，但是放钱之前要先选商品、现在还没放钱，所以出货之前也要先选商品
type HasItemState struct {
	vendingMachine *VendingMachine
}

func (h *HasItemState) requestItem() error {
	if h.vendingMachine.itemCount == 0 { // 没有商品了要置为无商品状态
		h.vendingMachine.setState(h.vendingMachine.noItem)
		return fmt.Errorf("there is not item, please add items")
	}
	fmt.Printf("item requested\n")
	h.vendingMachine.setState(h.vendingMachine.itemRequested)
	return nil
}
func (h *HasItemState) addItem(count int) error {
	fmt.Printf("%d items added", count)
	h.vendingMachine.incrementItemCount(count)
	return nil
}
func (h *HasItemState) insertMoney(money int) error {
	return fmt.Errorf("please select a item first")
}
func (h *HasItemState) dispenseItem() error {
	return fmt.Errorf("please select a item and insert money")
}

func StateFunc() {
	vendingMachine := newVendingMachine(0, 0)

	err := vendingMachine.requestItem()
	if err != nil {
		log.Fatal(err.Error())
	}

	err = vendingMachine.addItem(1)
	if err != nil {
		log.Fatal(err.Error())
	}

	err = vendingMachine.requestItem()
	if err != nil {
		log.Fatal(err.Error())
	}

}

```
### 扩展
- 添加状态就行了，但是添加一个状态会导致原来的状态机被破坏，害得修改原来的状态机代码