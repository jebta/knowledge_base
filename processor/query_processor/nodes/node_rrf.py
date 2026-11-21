from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.json_format import json_format
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")

        weight_embedding = [
            (embedding_chunks, 1),
            (hyde_embedding_chunks, 1)
        ]
        rrf_dict = {}
        for chunks, weight in weight_embedding:
            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.get("id")
                score = weight / (idx + 60)
                if chunk_id in rrf_dict:
                    rrf_dict[chunk_id]["score"] += score
                else:
                    chunk["score"] = score
                    rrf_dict[chunk_id] = chunk
        rrf_chunks = sorted(rrf_dict.values(), key=lambda x: x["score"], reverse=True)
        return {
            "rrf_chunks": rrf_chunks[:len(embedding_chunks)]
        }


if __name__ == '__main__':
    init_state = {
        "embedding_chunks": [
            {
                "id": 469167471379611937,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![图片展示一只手正接触设备内部高温部件，配文警告：设备使用后内部零件仍极热，切勿触摸灰色标记区域，以防烧伤；并强调需待冷却后再操作。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/71f9e17a760882436455f7ddc3a5fd358ce821ceb6e598b76d013d93dc26c943.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.827332079410553,
                "source": "local"
            },
            {
                "id": 469167471379611926,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟(中国)商业有限公司技术服务支持网站http://www.95105369.com/Web/Manuals.aspx上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
                "item_name": "HAK180烫金机",
                "score": 0.8250730633735657,
                "source": "local"
            },
            {
                "id": 469167471379611946,
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![正确搬运设备应双手握住机身两侧下方，避免抓握进纸托板或出纸盒；错误做法是手抓顶部托板，易致部件脱落。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/0bc44b278a939f7ec6ff239a4423c46a74a6e07572150f9e912f5fa55e0f598d.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.8249151110649109,
                "source": "local"
            },
            {
                "id": 469167471379611943,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![图片警示：禁止用手强行拉出设备部件（如进纸托板），应双手稳握设备底部搬运；同时提醒勿将手指置于危险区域，避免夹伤或设备损坏。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/3bf3b3639360a2c43c223a6ee46c44b27d54f4a79506a17a8b99e6ba4703df40.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.823890745639801,
                "source": "local"
            },
            {
                "id": 469167471379611931,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n\\- 请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n\\- 请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n\\- 请在以下环境使用本设备：温度保持在 $10^{\\circ}\\mathrm{C}$ 和 $32^{\\circ}\\mathrm{C}$ 之间，湿度保持在 $20\\%$ 和 $80\\%$ 之间，无冷凝。",
                "item_name": "HAK180烫金机",
                "score": 0.8209015727043152,
                "source": "local"
            },
            {
                "id": 469167471379611936,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![图片警示高温风险：设备内部部件（如图中灰色区域）工作后极热，禁止触摸以防烫伤；同时提醒勿将手伸入开放的设备内部，避免触电或火灾危险。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/37eecd8ac02745dac415a628b51c26974067062c9b9747c069579fdaa99dbae9.jpg)  \n高温图标警告您切勿触摸高温的设备部件。",
                "item_name": "HAK180烫金机",
                "score": 0.818871021270752,
                "source": "local"
            },
            {
                "id": 469167471379611941,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n\\- 将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n\\- 将本设备放置在通风良好的环境中。\n\n\\- 为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。",
                "item_name": "HAK180烫金机",
                "score": 0.7099866271018982,
                "source": "local"
            },
            {
                "id": 469167471379611942,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n\\- 为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![图片警示禁止将手指伸入设备内部的齿轮或传动部件区域，以防夹伤；强调操作时需避免手部接触图示危险部位，确保人身安全。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/944ad0b1ae642d4a65bc1b40c979beeb12bc30e6a409b9d9666ee013ff48cc60.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.7068914771080017,
                "source": "local"
            },
            {
                "id": 469167471379611927,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n\\- 对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n\\- 如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n\\- 本文档中提供的信息可能会随时更改，恕不另行通知。\n\n\\- 严禁未经授权擅自复制或重制本文档的任何部分或全部内容。",
                "item_name": "HAK180烫金机",
                "score": 0.7064312100410461,
                "source": "local"
            },
            {
                "id": 469167471379611948,
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![图示禁止将设备放置于桌面边缘，因出纸盒或进纸托板可能脱落；应确保设备完全置于平整稳定台面，避免悬空，以防跌落造成人身伤害或设备损坏。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/baa3590221a744dda3b19dfb126ddc6a3fd79b59cf19ff8d6793b51114c0d57b.jpg)\n\n“重要事项”表示可能导致财产损失或本设备功能丧失的潜在危险情况。",
                "item_name": "HAK180烫金机",
                "score": 0.6997081637382507,
                "source": "local"
            }
        ],
        "hyde_embedding_chunks": [
            {
                "id": 469167471379611928,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n\\- 严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n\\- 请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。",
                "item_name": "HAK180烫金机",
                "score": 0.8367994427680969,
                "source": "local"
            },
            {
                "id": 469167471379611937,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![图片展示一只手正接触设备内部高温部件，配文警告：设备使用后内部零件仍极热，切勿触摸灰色标记区域，以防烧伤；并强调需待冷却后再操作。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/71f9e17a760882436455f7ddc3a5fd358ce821ceb6e598b76d013d93dc26c943.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.8273680806159973,
                "source": "local"
            },
            {
                "id": 469167471379611943,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![图片警示：禁止用手强行拉出设备部件（如进纸托板），应双手稳握设备底部搬运；同时提醒勿将手指置于危险区域，避免夹伤或设备损坏。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/3bf3b3639360a2c43c223a6ee46c44b27d54f4a79506a17a8b99e6ba4703df40.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.8200389742851257,
                "source": "local"
            },
            {
                "id": 469167471379611926,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟(中国)商业有限公司技术服务支持网站http://www.95105369.com/Web/Manuals.aspx上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
                "item_name": "HAK180烫金机",
                "score": 0.8147937655448914,
                "source": "local"
            },
            {
                "id": 469167471379611936,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![图片警示高温风险：设备内部部件（如图中灰色区域）工作后极热，禁止触摸以防烫伤；同时提醒勿将手伸入开放的设备内部，避免触电或火灾危险。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/37eecd8ac02745dac415a628b51c26974067062c9b9747c069579fdaa99dbae9.jpg)  \n高温图标警告您切勿触摸高温的设备部件。",
                "item_name": "HAK180烫金机",
                "score": 0.8102648854255676,
                "source": "local"
            },
            {
                "id": 469167471379611946,
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![正确搬运设备应双手握住机身两侧下方，避免抓握进纸托板或出纸盒；错误做法是手抓顶部托板，易致部件脱落。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/0bc44b278a939f7ec6ff239a4423c46a74a6e07572150f9e912f5fa55e0f598d.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.7015323638916016,
                "source": "local"
            },
            {
                "id": 469167471379611948,
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![图示禁止将设备放置于桌面边缘，因出纸盒或进纸托板可能脱落；应确保设备完全置于平整稳定台面，避免悬空，以防跌落造成人身伤害或设备损坏。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/baa3590221a744dda3b19dfb126ddc6a3fd79b59cf19ff8d6793b51114c0d57b.jpg)\n\n“重要事项”表示可能导致财产损失或本设备功能丧失的潜在危险情况。",
                "item_name": "HAK180烫金机",
                "score": 0.6987366676330566,
                "source": "local"
            },
            {
                "id": 469167471379611927,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n\\- 对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n\\- 如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n\\- 本文档中提供的信息可能会随时更改，恕不另行通知。\n\n\\- 严禁未经授权擅自复制或重制本文档的任何部分或全部内容。",
                "item_name": "HAK180烫金机",
                "score": 0.6970774531364441,
                "source": "local"
            },
            {
                "id": 469167471379611941,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n\\- 将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n\\- 将本设备放置在通风良好的环境中。\n\n\\- 为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。",
                "item_name": "HAK180烫金机",
                "score": 0.6957339644432068,
                "source": "local"
            },
            {
                "id": 469167471379611942,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n\\- 为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![图片警示禁止将手指伸入设备内部的齿轮或传动部件区域，以防夹伤；强调操作时需避免手部接触图示危险部位，确保人身安全。](http://192.168.100.100:9000/knowledge-base/upload-images/20260918/hak180产品安全手册/944ad0b1ae642d4a65bc1b40c979beeb12bc30e6a409b9d9666ee013ff48cc60.jpg)",
                "item_name": "HAK180烫金机",
                "score": 0.6937724947929382,
                "source": "local"
            }
        ]
    }
    node_rrf = NodeRrf()
    res = node_rrf(init_state)
    logger.info(json_format(res))
