### template method模式是什么
- 在超类中定义了算法框架，允许子类在不改变结构的额情况下重写算法的特定步骤
- 基于继承、是静态的
### 适用场景
- 
### 实现
#### 概念
- 
#### 具体实现
- 将一个一次性密码功能（OTP）传递给用户，发送方式有很多种，比如短信，邮件，但是整个OTP流程是一样的
- 定义OTP接口：包含随机生成OTP密码、保存OTP到缓存、获得信息、发送通知方法
- 定义发送主体：包含一个OPT接口，以及整个操作流程
- 具体的OTP方法：实现接口
#### c++实现
```c++

```
#### go实现
```go
package templatemethod

import (
	"fmt"
	// "github.com/go-playground/locales/ms"
)

// 定义OTP接口
type IOtp interface {
	genRandomOtp(int) string
	saveToCache(string)
	getMessage(string) string
	sendNotification(string) error
}

// 定义一个实例
type Otp struct {
	iOtp IOtp
}

// 实例完成操作的流程
func (o *Otp) genAndSendOtp(otpLen int) error {
	otp := o.iOtp.genRandomOtp(otpLen)
	o.iOtp.saveToCache(otp)
	msg := o.iOtp.getMessage(otp)
	err := o.iOtp.sendNotification(msg)
	if err != nil {
		return err
	}
	return nil
}

// 定义具体实施的类型
type Sms struct {
	Otp
}

func (s *Sms) genRandomOtp(len int) string {
	randomOTP := "1234"
	fmt.Printf("SMS:generating random otp: %s\n", randomOTP)
	return randomOTP
}

func (s *Sms) saveToCache(otp string) {
	fmt.Printf("SMS: saving otp:%s", otp)
}

func (s *Sms) getMessage(Otp string) string {
	return "SMS OTP for login is " + Otp
}

func (s *Sms) sendNotification(msg string) error {
	fmt.Printf("SMS: sending sms: %s\n", msg)
	return nil
}

func TemplateMethod() {
	smsOtp := &Sms{}
	o := Otp{
		iOtp: smsOtp,
	}
	o.genAndSendOtp(4)
}

```
### 扩展
- 添加方法具体的发送者就行，这在go实现起来比较简单，就是接口实现