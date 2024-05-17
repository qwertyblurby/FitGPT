import Floater from './Floater';

function SectionTech() {
	const floaters = [
		[
			"Model Architecture",
			"Our AI model is built using a state-of-the-art deep learning architecture, leveraging the latest advancements in computer vision.",
			"TypeIcon"
		],
		[
			"Training Dataset",
			"The model has been trained on a diverse and comprehensive dataset, ensuring robust performance across a wide range of image categories.",
			"TrainTrackIcon"
		],
		[
			"Inference Speed",
			"Our AI model is optimized for lightning-fast inference, processing images in milliseconds and delivering real-time results.",
			"InfoIcon"
		]
	]
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Technical Details</h2>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
				{floaters.map(([titleText, description, iconSrc]) => (
					<Floater
						key={titleText}
						titleText={titleText}
						description={description}
						iconSrc={iconSrc}
						boxBg={true}
					/>
				))}
			</div>
		</div>
	)
}

export default SectionTech;
