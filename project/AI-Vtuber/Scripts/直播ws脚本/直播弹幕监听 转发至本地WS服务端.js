// ==UserScript==
// @name         直播弹幕监听 转发至本地WS服务端
// @namespace    http://tampermonkey.net/
// @version      0.9
// @description  观察指定 DOM 节点的变化以将数据发送到连接的WebSocket服务端
// @description  Github：https://github.com/Ikaros-521/AI-Vtuber/tree/main/Scripts/%E7%9B%B4%E6%92%ADws%E8%84%9A%E6%9C%AC
// @author       Ikaros
// @match        https://www.douyu.com/*
// @match        https://live.kuaishou.com/u/*
// @match        https://live.kuaishou.com/u/*
// @match        https://mobile.yangkeduo.com/*
// @match        https://live.1688.com/zb/play.html*
// @grant        none
// @namespace    https://greasyfork.org/scripts/490966
// @license      GPL-3.0
// @downloadURL https://update.greasyfork.org/scripts/490966/%E7%9B%B4%E6%92%AD%E5%BC%B9%E5%B9%95%E7%9B%91%E5%90%AC%20%E8%BD%AC%E5%8F%91%E8%87%B3%E6%9C%AC%E5%9C%B0WS%E6%9C%8D%E5%8A%A1%E7%AB%AF.user.js
// @updateURL https://update.greasyfork.org/scripts/490966/%E7%9B%B4%E6%92%AD%E5%BC%B9%E5%B9%95%E7%9B%91%E5%90%AC%20%E8%BD%AC%E5%8F%91%E8%87%B3%E6%9C%AC%E5%9C%B0WS%E6%9C%8D%E5%8A%A1%E7%AB%AF.meta.js
// ==/UserScript==

(function () {
  "use strict";

  setTimeout(function() {
    let my_socket = null;
    let wsUrl = "ws://127.0.0.1:5000";
    let targetNode = null;
    let my_observer = null;

    const hostname = window.location.hostname;

    if (hostname === "www.douyu.com") {
      console.log("当前直播平台：斗鱼");
      wsUrl = "ws://127.0.0.1:5000";
    } else if (hostname === "live.kuaishou.com") {
      console.log("当前直播平台：快手");
      wsUrl = "ws://127.0.0.1:5000";
    } else if (hostname === "mobile.yangkeduo.com") {
      console.log("当前直播平台：拼多多");
      wsUrl = "ws://127.0.0.1:5000";
    } else if (hostname === "live.1688.com") {
      console.log("当前直播平台：1688");
      wsUrl = "ws://127.0.0.1:5000";
    }

    function connectWebSocket() {
      // 创建 WebSocket 连接，适配服务端
      my_socket = new WebSocket(wsUrl);

      // 当连接建立时触发
      my_socket.addEventListener("open", (event) => {
        console.log("ws连接打开");

        // 向服务器发送一条消息
        const data = {
          type: "info",
          content: "ws连接成功",
        };
        console.log(data);
        my_socket.send(JSON.stringify(data));
      });

      // 当收到消息时触发
      my_socket.addEventListener("message", (event) => {
        console.log("收到服务器数据:", event.data);
      });

      // 当连接关闭时触发
      my_socket.addEventListener("close", (event) => {
        console.log("WS连接关闭");

        // 重连
        setTimeout(() => {
          connectWebSocket();
        }, 1000); // 延迟 1 秒后重连
      });
    }

    // 初始连接
    connectWebSocket();
    if (hostname === "www.douyu.com") {
      // 选择需要观察变化的节点
      targetNode = document.querySelector(".Barrage-list");

      // 创建观察器实例
      my_observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          // 这里处理新增的DOM元素
          if (mutation.type === "childList") {
            mutation.addedNodes.forEach((node) => {
              // 判断是否是新增的弹幕消息
              if (node.classList.contains("Barrage-listItem")) {
                // 新增的动态DOM元素处理
                // console.log('Added node:', node);

                const spans = node.getElementsByTagName("span");

                let username = "";
                let content = "";

                for (let span of spans) {
                  //console.log(span);
                  if (span.classList.contains("Barrage-nickName")) {
                    const targetSpan = span;
                    // 获取用户名
                    let tmp = targetSpan.textContent.trim().slice(0, -1);
                    if (tmp != "")
                      username = targetSpan.textContent.trim().slice(0, -1);
                  } else if (span.classList.contains("Barrage-content")) {
                    const targetSpan = span;
                    // 获取弹幕内容
                    content = targetSpan.textContent.trim();
                  }
                }

                console.log(username + ":" + content);

                // 获取到弹幕数据
                if (username != "" && content != "") {
                  const data = {
                    type: "comment",
                    username: username,
                    content: content,
                  };
                  console.log(data);
                  my_socket.send(JSON.stringify(data));
                }
              }
            });
          }
        });
      });
    } else if (hostname === "live.kuaishou.com") {
      // 选择需要观察变化的节点
      targetNode = document.querySelector(".chat-history");

      // 创建观察器实例
      my_observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          // 这里处理新增的DOM元素
          if (mutation.type === "childList") {
            mutation.addedNodes.forEach((node) => {
              // 判断是否是新增的弹幕消息
              if (node.classList.contains("chat-info")) {
                // 新增的动态DOM元素处理
                console.log("Added node:", node);

                const usernameElement = node.querySelector(".username");
                const commentElement = node.querySelector(".comment");

                // 礼物数据
                const giftCommentElement = node.querySelector(".gift-comment");
                const giftImgElement = node.querySelector(".gift-img");

                const likeElement = node.querySelector(".like");

                if (usernameElement && giftCommentElement) {
                  // 礼物数据处理
                  const username = usernameElement.textContent.trim();
                  console.log(username + "送出了礼物");

                  // 如果 my_socket 已经初始化，可以在这里发送礼物数据
                  if (my_socket) {
                    const data = {
                      type: "gift",
                      username: username,
                      // 可以根据需要添加其他礼物相关数据
                    };
                    console.log(data);
                    my_socket.send(JSON.stringify(data));
                  }
                } else if (usernameElement && likeElement) {
                  const username = usernameElement.textContent.trim();
                  console.log(username + "点了个赞");

                  // 如果 my_socket 已经初始化，可以在这里发送礼物数据
                  if (my_socket) {
                    const data = {
                      type: "like",
                      username: username,
                      // 可以根据需要添加其他礼物相关数据
                    };
                    console.log(data);
                    my_socket.send(JSON.stringify(data));
                  }
                } else if (
                  usernameElement &&
                  commentElement &&
                  !giftCommentElement &&
                  !likeElement
                ) {
                  const username = usernameElement.textContent.trim().slice(0, -1);
                  const content = commentElement.textContent.trim();

                  console.log(username + ":" + content);

                  // 获取到弹幕数据
                  if (username !== "" && content !== "") {
                    const data = {
                      type: "comment",
                      username: username,
                      content: content,
                    };
                    console.log(data);
                    // 如果 my_socket 已经初始化，可以在这里发送数据
                    if (my_socket) {
                      my_socket.send(JSON.stringify(data));
                    }
                  }
                }
              }
            });
          }
        });
      });
    } else if (hostname === "mobile.yangkeduo.com") {
      // 选择需要观察变化的节点
      targetNode = document.querySelector(".MYFlHgGu");

      // 创建观察器实例
      my_observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          // 这里处理新增的DOM元素
          if (mutation.type === "childList") {
            mutation.addedNodes.forEach((node) => {
              // 判断是否是新增的弹幕消息
              if (node.classList.contains("_24Qh0Jmi")) {
                // 新增的动态DOM元素处理
                console.log("Added node:", node);

                const usernameElement = node.querySelector(".t6fCgSnz");
                const commentElement = node.querySelector("._16_fPXYP");

                if (
                  usernameElement &&
                  commentElement
                ) {
                  const username = usernameElement.textContent.trim().slice(0, -1);
                  const content = commentElement.textContent.trim();

                  console.log(username + ":" + content);

                  // 获取到弹幕数据
                  if (username !== "" && content !== "") {
                    const data = {
                      type: "comment",
                      username: username,
                      content: content,
                    };
                    console.log(data);
                    // 如果 my_socket 已经初始化，可以在这里发送数据
                    if (my_socket) {
                      my_socket.send(JSON.stringify(data));
                    }
                  }
                }
              }
            });
          }
        });
      });
    } else if (hostname === "live.1688.com") {
      // 选择需要观察变化的节点
      targetNode = document.querySelector(".pc-living-room-message");

      // 创建观察器实例
      my_observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          // 这里处理新增的DOM元素
          if (mutation.type === "childList") {
            mutation.addedNodes.forEach((node) => {
              // 判断是否是新增的弹幕消息
              if (node.classList.contains("comment-message")) {
                // 新增的动态DOM元素处理
                console.log("Added node:", node);

                const usernameElement = node.querySelector(".from");
                const commentElement = node.querySelector(".msg-text");

                if (
                  usernameElement &&
                  commentElement
                ) {
                  const username = usernameElement.textContent.trim().slice(0, -1);
                  const content = commentElement.textContent.trim();

                  console.log(username + ":" + content);

                  // 获取到弹幕数据
                  if (username !== "" && content !== "") {
                    const data = {
                      type: "comment",
                      username: username,
                      content: content,
                    };
                    console.log(data);
                    // 如果 my_socket 已经初始化，可以在这里发送数据
                    if (my_socket) {
                      my_socket.send(JSON.stringify(data));
                    }
                  }
                }
              }
            });
          }
        });
      });
    }

    // 配置观察选项
    const config = {
      childList: true,
      subtree: true,
    };

    // 开始观察
    my_observer.observe(targetNode, config);
  }, 3000);
})();
